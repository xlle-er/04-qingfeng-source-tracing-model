"""Retrain the Beijing three-station PM2.5 forecasting models offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.lstm_predictor import LSTMPredictor


FEATURES = ["pm25", "pm25_天坛", "pm25_永定门内"]
SEEDS = [42, 43, 44]
BOUNDARY = pd.Timestamp("2019-03-14", tz="Asia/Shanghai")


def windows(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
    part = frame.loc[(frame.index >= start) & (frame.index < end), FEATURES]
    values = part.to_numpy(dtype=np.float32)
    X, y = [], []
    for index in range(24, len(part) - 5):
        history = values[index - 24 : index]
        target = values[index : index + 6, 0]
        if np.isfinite(history).all() and np.isfinite(target).all():
            X.append(history)
            y.append(target)
    return np.asarray(X), np.asarray(y)


def predict_array(model: LSTMPredictor, X: np.ndarray) -> np.ndarray:
    model.model.eval()
    scaled = torch.from_numpy(np.ascontiguousarray(model._scale_X(X))).to(model.device)
    with torch.no_grad():
        prediction = model._inverse_y(model.model(scaled).cpu().numpy())
    return np.maximum(prediction, 0)


def score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": 1.0 - float(np.sum(error**2)) / denominator if denominator else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/beijing_neighborhood_2019/model_features.csv",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "output/beijing_retrained")
    args = parser.parse_args()

    frame = pd.read_csv(args.data, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    if list(frame.columns) != FEATURES:
        raise ValueError(f"Unexpected feature columns: {list(frame.columns)}")
    if frame.index.tz is None:
        raise ValueError("Input timestamps must include a timezone")

    start = frame.index.min()
    end = frame.index.max() + pd.Timedelta(hours=1)
    X_dev, y_dev = windows(frame, start, BOUNDARY)
    X_test, y_test = windows(frame, BOUNDARY, end)
    if len(X_dev) < 150 or len(X_test) < 30:
        raise ValueError("Insufficient complete windows")

    args.output.mkdir(parents=True, exist_ok=True)
    results = {"development_windows": len(X_dev), "test_windows": len(X_test), "models": {}}
    predictions: dict[str, np.ndarray] = {}
    torch.set_num_threads(2)

    for name, feature_count in [("single_station", 1), ("multi_station", 3)]:
        members = []
        seed_scores = []
        for seed in SEEDS:
            model = LSTMPredictor(
                input_size=feature_count,
                hidden_size=32,
                output_size=6,
                device=torch.device("cpu"),
                random_seed=seed,
            )
            model.feature_columns = FEATURES[:feature_count]
            model.target_column = "pm25"
            history = model.train(X_dev[:, :, :feature_count], y_dev, epochs=80, batch_size=64, patience=10)
            prediction = predict_array(model, X_test[:, :, :feature_count])
            members.append(prediction)
            seed_scores.append({"seed": seed, **score(y_test, prediction), "best_epoch": history["best_epoch"]})
            model.save(str(args.output / f"{name}_{seed}.pt"))
        ensemble = np.mean(members, axis=0)
        predictions[name] = ensemble
        results["models"][name] = {"ensemble": score(y_test, ensemble), "members": seed_scores}

    persistence = np.repeat(X_test[:, -1, 0:1], 6, axis=1)
    previous_day = X_test[:, :6, 0]
    results["baselines"] = {
        "persistence": score(y_test, persistence),
        "previous_day": score(y_test, previous_day),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
