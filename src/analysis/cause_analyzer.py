"""LSTM 污染高值事件成因分类。

该模块只在有人工标注的真实事件上训练和报告指标。没有标注数据时，
不会伪造“准确率≥80%”。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.models.lstm_predictor import get_device


DEFAULT_CAUSES = ["工业排放", "机动车尾气", "扬尘", "燃烧源", "二次生成", "气象累积"]


class PollutionCauseLSTM(nn.Module):
    def __init__(self, input_size: int, class_count: int, hidden_size: int = 64):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=2,
            dropout=0.2,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, class_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.encoder(x)
        return self.classifier(sequence[:, -1, :])


class CauseAnalyzer:
    """面向竞赛评估的时序成因分类器。"""

    def __init__(
        self,
        input_size: int,
        class_names: Optional[List[str]] = None,
        hidden_size: int = 64,
        learning_rate: float = 1e-3,
        device: Optional[torch.device] = None,
        random_seed: int = 42,
    ):
        self.class_names = class_names or list(DEFAULT_CAUSES)
        if len(self.class_names) < 2 or len(set(self.class_names)) != len(self.class_names):
            raise ValueError("成因类别至少两类且不能重复")
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        self.input_size = input_size
        self.device = device or get_device()
        self.model = PollutionCauseLSTM(input_size, len(self.class_names), hidden_size).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.feature_columns: Optional[List[str]] = None
        self.seq_length = 24
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None
        self.is_fitted = False

    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        label_column: str = "cause_label",
        seq_length: int = 24,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """以标注时刻前的窗口作为样本，避免引入事件后信息。"""
        missing = [c for c in [*feature_columns, label_column] if c not in df.columns]
        if missing:
            raise KeyError(f"缺少列: {missing}")
        if len(feature_columns) != self.input_size:
            raise ValueError("特征数与 input_size 不一致")
        features = df[feature_columns].to_numpy(dtype=np.float32, copy=True)
        mapping = {name: idx for idx, name in enumerate(self.class_names)}
        X, y = [], []
        for end in range(seq_length - 1, len(df)):
            label = df.iloc[end][label_column]
            if pd.isna(label):
                continue
            if label not in mapping:
                raise ValueError(f"未知成因标签: {label}")
            window = features[end - seq_length + 1 : end + 1]
            if np.isfinite(window).all():
                X.append(window)
                y.append(mapping[label])
        self.feature_columns = list(feature_columns)
        self.seq_length = seq_length
        if not X:
            return np.empty((0, seq_length, self.input_size), np.float32), np.empty(0, np.int64)
        return np.ascontiguousarray(X), np.asarray(y, dtype=np.int64)

    def _scale(self, X: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_std is None:
            return X.astype(np.float32, copy=True)
        return ((X - self.feature_mean) / self.feature_std).astype(np.float32)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 80,
        batch_size: int = 32,
        validation_split: float = 0.2,
        patience: int = 10,
    ) -> Dict[str, object]:
        """按时间顺序切分，不随机打散高度重叠的时序窗口。"""
        if len(X) < 4 or len(X) != len(y):
            raise ValueError("至少需要4个有标注样本，且 X/y 样本数相同")
        split = min(max(int(len(X) * (1 - validation_split)), 1), len(X) - 1)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        flat = X_train.reshape(-1, X_train.shape[-1])
        self.feature_mean = flat.mean(axis=0, keepdims=True).astype(np.float32)
        self.feature_std = flat.std(axis=0, keepdims=True).astype(np.float32)
        self.feature_std[self.feature_std < 1e-6] = 1.0

        counts = np.bincount(y_train, minlength=len(self.class_names)).astype(np.float32)
        weights = np.zeros_like(counts)
        present = counts > 0
        weights[present] = len(y_train) / (present.sum() * counts[present])
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(self.device))
        train_X = torch.from_numpy(np.ascontiguousarray(self._scale(X_train))).to(self.device)
        val_X = torch.from_numpy(np.ascontiguousarray(self._scale(X_val))).to(self.device)
        train_y = torch.from_numpy(y_train).to(self.device)
        val_y = torch.from_numpy(y_val).to(self.device)

        history: Dict[str, object] = {"train_loss": [], "val_loss": [], "best_epoch": 0}
        best_loss, stale = float("inf"), 0
        best_state = deepcopy(self.model.state_dict())
        for epoch in range(epochs):
            self.model.train()
            total, seen = 0.0, 0
            for begin in range(0, len(train_X), batch_size):
                bx, by = train_X[begin : begin + batch_size], train_y[begin : begin + batch_size]
                self.optimizer.zero_grad()
                loss = criterion(self.model(bx), by)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total += loss.item() * len(bx)
                seen += len(bx)
            self.model.eval()
            with torch.no_grad():
                val_loss = float(criterion(self.model(val_X), val_y).item())
            history["train_loss"].append(total / max(seen, 1))
            history["val_loss"].append(val_loss)
            if val_loss < best_loss - 1e-4:
                best_loss, stale = val_loss, 0
                best_state = deepcopy(self.model.state_dict())
                history["best_epoch"] = epoch + 1
            else:
                stale += 1
                if patience > 0 and stale >= patience:
                    break
        self.model.load_state_dict(best_state)
        self.is_fitted = True
        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("成因分类器尚未在标注数据上训练，禁止输出随机初始化结果")
        self.model.eval()
        tensor = torch.from_numpy(np.ascontiguousarray(self._scale(X))).to(self.device)
        with torch.no_grad():
            return torch.softmax(self.model(tensor), dim=1).cpu().numpy()

    def predict(self, X: np.ndarray) -> List[Dict[str, object]]:
        probabilities = self.predict_proba(X)
        results = []
        for row in probabilities:
            order = np.argsort(row)[::-1]
            results.append(
                {
                    "cause": self.class_names[int(order[0])],
                    "confidence": float(row[order[0]]),
                    "probabilities": {self.class_names[int(i)]: float(row[i]) for i in order},
                }
            )
        return results

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, object]:
        pred = self.predict_proba(X).argmax(axis=1)
        class_count = len(self.class_names)
        confusion = np.zeros((class_count, class_count), dtype=int)
        for actual, guessed in zip(y, pred):
            confusion[int(actual), int(guessed)] += 1
        accuracy = float(np.mean(pred == y)) if len(y) else 0.0
        f1_scores, recalls = [], {}
        for idx, name in enumerate(self.class_names):
            tp = confusion[idx, idx]
            fp = confusion[:, idx].sum() - tp
            fn = confusion[idx, :].sum() - tp
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1_scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
            recalls[name] = float(recall)
        return {
            "accuracy": accuracy,
            "macro_f1": float(np.mean(f1_scores)),
            "per_class_recall": recalls,
            "confusion_matrix": confusion.tolist(),
            "sample_count": int(len(y)),
        }

    def save(self, path: str) -> None:
        """保存模型结构、类别、特征顺序和训练集标准化参数。"""
        if not self.is_fitted:
            raise RuntimeError("成因分类器尚未训练，不能保存")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "input_size": self.input_size,
                "hidden_size": int(self.model.encoder.hidden_size),
                "class_names": self.class_names,
                "feature_columns": self.feature_columns,
                "seq_length": self.seq_length,
                "feature_mean": self.feature_mean,
                "feature_std": self.feature_std,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str,
        device: Optional[torch.device] = None,
    ) -> "CauseAnalyzer":
        """恢复训练后的成因分类器和预处理参数。"""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        analyzer = cls(
            input_size=int(checkpoint["input_size"]),
            hidden_size=int(checkpoint["hidden_size"]),
            class_names=list(checkpoint["class_names"]),
            device=device,
        )
        analyzer.model.load_state_dict(checkpoint["model_state_dict"])
        analyzer.feature_columns = list(checkpoint.get("feature_columns") or [])
        analyzer.seq_length = int(checkpoint.get("seq_length", 24))
        analyzer.feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        analyzer.feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
        analyzer.is_fitted = True
        return analyzer
