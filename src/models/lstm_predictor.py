"""可复现、无时序泄漏的 LSTM 多步预测器。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def get_device() -> torch.device:
    """获取可用计算设备（MPS > CUDA > CPU）。"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(x)
        return self.fc(sequence[:, -1, :])


class LSTMPredictor:
    """包含训练集标准化、早停、梯度裁剪和评估指标的预测器。"""

    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 64,
        output_size: int = 6,
        learning_rate: float = 0.001,
        device: Optional[torch.device] = None,
        random_seed: int = 42,
    ):
        if min(input_size, hidden_size, output_size) <= 0:
            raise ValueError("模型维度必须为正整数")
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device or get_device()
        self.model = LSTMModel(input_size, hidden_size, output_size).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.feature_columns: Optional[List[str]] = None
        self.target_column: Optional[str] = None
        self.seq_length = 24
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None
        self.target_mean: Optional[np.ndarray] = None
        self.target_std: Optional[np.ndarray] = None
        self.is_fitted = False

    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
        seq_length: int = 24,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if seq_length <= 0:
            raise ValueError("seq_length 必须大于0")
        missing = [c for c in [*feature_columns, target_column] if c not in df.columns]
        if missing:
            raise KeyError(f"缺少列: {missing}")
        if len(feature_columns) != self.input_size:
            raise ValueError(f"特征数 {len(feature_columns)} 与 input_size={self.input_size} 不一致")

        data = df[feature_columns].to_numpy(dtype=np.float32, copy=True)
        target = df[target_column].to_numpy(dtype=np.float32, copy=True)
        if not np.isfinite(data).all() or not np.isfinite(target).all():
            raise ValueError("训练数据包含 NaN 或无穷值")

        sample_count = len(data) - seq_length - self.output_size + 1
        if sample_count <= 0:
            return (
                np.empty((0, seq_length, self.input_size), dtype=np.float32),
                np.empty((0, self.output_size), dtype=np.float32),
            )
        X = np.stack([data[i : i + seq_length] for i in range(sample_count)])
        y = np.stack(
            [target[i + seq_length : i + seq_length + self.output_size] for i in range(sample_count)]
        )
        self.feature_columns = list(feature_columns)
        self.target_column = target_column
        self.seq_length = seq_length
        return np.ascontiguousarray(X), np.ascontiguousarray(y)

    def _fit_scalers(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        flat = X_train.reshape(-1, X_train.shape[-1])
        self.feature_mean = flat.mean(axis=0, keepdims=True).astype(np.float32)
        self.feature_std = flat.std(axis=0, keepdims=True).astype(np.float32)
        self.feature_std[self.feature_std < 1e-6] = 1.0
        self.target_mean = y_train.mean(axis=0, keepdims=True).astype(np.float32)
        self.target_std = y_train.std(axis=0, keepdims=True).astype(np.float32)
        self.target_std[self.target_std < 1e-6] = 1.0

    def _scale_X(self, X: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_std is None:
            return X.astype(np.float32, copy=True)
        return ((X - self.feature_mean) / self.feature_std).astype(np.float32)

    def _scale_y(self, y: np.ndarray) -> np.ndarray:
        if self.target_mean is None or self.target_std is None:
            return y.astype(np.float32, copy=True)
        return ((y - self.target_mean) / self.target_std).astype(np.float32)

    def _inverse_y(self, y: np.ndarray) -> np.ndarray:
        if self.target_mean is None or self.target_std is None:
            return y
        return y * self.target_std + self.target_mean

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        patience: int = 10,
        min_delta: float = 1e-4,
        purge_gap: bool = True,
    ) -> Dict[str, object]:
        if X.ndim != 3 or y.ndim != 2 or len(X) != len(y):
            raise ValueError("X/y 形状应为 (N,T,F) 和 (N,H)，且样本数相同")
        if len(X) < 2:
            raise ValueError("至少需要2个时序样本")
        if not 0 < validation_split < 1:
            raise ValueError("validation_split 必须在 (0,1) 内")
        split = int(len(X) * (1 - validation_split))
        split = min(max(split, 1), len(X) - 1)
        # 相邻滑窗高度重叠。在训练/验证之间清除一个完整输入+输出窗口，
        # 避免同一时段同时出现在两侧造成乐观评估。
        requested_gap = X.shape[1] + y.shape[1] - 1 if purge_gap else 0
        gap = requested_gap if split - requested_gap >= 1 else 0
        train_end = split - gap
        X_train_raw, X_val_raw = X[:train_end], X[split:]
        y_train_raw, y_val_raw = y[:train_end], y[split:]
        self._fit_scalers(X_train_raw, y_train_raw)

        X_train = torch.from_numpy(np.ascontiguousarray(self._scale_X(X_train_raw))).to(self.device)
        X_val = torch.from_numpy(np.ascontiguousarray(self._scale_X(X_val_raw))).to(self.device)
        y_train = torch.from_numpy(np.ascontiguousarray(self._scale_y(y_train_raw))).to(self.device)
        y_val = torch.from_numpy(np.ascontiguousarray(self._scale_y(y_val_raw))).to(self.device)

        history: Dict[str, object] = {
            "train_loss": [],
            "val_loss": [],
            "best_epoch": 0,
            "stopped_early": False,
            "purge_gap_samples": gap,
        }
        best_loss = float("inf")
        best_state = deepcopy(self.model.state_dict())
        stale = 0

        for epoch in range(epochs):
            self.model.train()
            loss_sum = 0.0
            seen = 0
            for begin in range(0, len(X_train), batch_size):
                batch_X = X_train[begin : begin + batch_size]
                batch_y = y_train[begin : begin + batch_size]
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(batch_X), batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                loss_sum += loss.item() * len(batch_X)
                seen += len(batch_X)

            self.model.eval()
            with torch.no_grad():
                val_loss = float(self.criterion(self.model(X_val), y_val).item())
            history["train_loss"].append(loss_sum / max(seen, 1))
            history["val_loss"].append(val_loss)
            if val_loss < best_loss - min_delta:
                best_loss = val_loss
                best_state = deepcopy(self.model.state_dict())
                history["best_epoch"] = epoch + 1
                stale = 0
            else:
                stale += 1
                if patience > 0 and stale >= patience:
                    history["stopped_early"] = True
                    break

        self.model.load_state_dict(best_state)
        self.is_fitted = True
        return history

    def predict(self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> np.ndarray:
        columns = feature_columns or self.feature_columns
        if columns is None:
            columns = list(df.select_dtypes(include=[np.number]).columns)
        if len(columns) != self.input_size:
            raise ValueError(f"推理特征数 {len(columns)} 与 input_size={self.input_size} 不一致")
        if len(df) < self.seq_length:
            raise ValueError(f"推理至少需要 {self.seq_length} 条记录")
        values = df[columns].tail(self.seq_length).to_numpy(dtype=np.float32, copy=True)
        if not np.isfinite(values).all():
            raise ValueError("推理数据包含 NaN 或无穷值")
        scaled = self._scale_X(values[np.newaxis, ...])
        self.model.eval()
        with torch.no_grad():
            output = self.model(torch.from_numpy(np.ascontiguousarray(scaled)).to(self.device)).cpu().numpy()
        prediction = self._inverse_y(output).reshape(-1)
        return np.clip(prediction, 0, None)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        if len(X) == 0:
            raise ValueError("评估集不能为空")
        self.model.eval()
        scaled = torch.from_numpy(np.ascontiguousarray(self._scale_X(X))).to(self.device)
        with torch.no_grad():
            pred = self.model(scaled).cpu().numpy()
        pred = self._inverse_y(pred)
        error = pred - y
        mae = float(np.mean(np.abs(error)))
        rmse = float(np.sqrt(np.mean(error**2)))
        denom = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - float(np.sum(error**2)) / denom if denom > 0 else 0.0
        return {"mae": mae, "rmse": rmse, "r2": r2}

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "output_size": self.output_size,
                "feature_columns": self.feature_columns,
                "target_column": self.target_column,
                "seq_length": self.seq_length,
                "feature_mean": self.feature_mean,
                "feature_std": self.feature_std,
                "target_mean": self.target_mean,
                "target_std": self.target_std,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str,
        device: Optional[torch.device] = None,
    ) -> "LSTMPredictor":
        """恢复模型、特征顺序与训练集标准化参数。"""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        predictor = cls(
            input_size=int(checkpoint["input_size"]),
            hidden_size=int(checkpoint["hidden_size"]),
            output_size=int(checkpoint["output_size"]),
            device=device,
        )
        predictor.model.load_state_dict(checkpoint["model_state_dict"])
        predictor.feature_columns = checkpoint.get("feature_columns") or checkpoint.get("feature_cols")
        predictor.target_column = checkpoint.get("target_column", "pm25")
        predictor.seq_length = int(checkpoint.get("seq_length", 24))
        for name in ("feature_mean", "feature_std", "target_mean", "target_std"):
            value = checkpoint.get(name)
            setattr(predictor, name, None if value is None else np.asarray(value, dtype=np.float32))
        predictor.is_fitted = predictor.feature_mean is not None and predictor.target_mean is not None
        return predictor
