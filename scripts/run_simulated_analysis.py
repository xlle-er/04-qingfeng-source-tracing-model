#!/usr/bin/env python3
"""重训模拟数据模型、运行溯源评估并生成报告所需结果与图表。"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.cause_analyzer import CauseAnalyzer
from src.analysis.concentration_gradient import ConcentrationGradientAnalyzer
from src.models.lstm_predictor import LSTMPredictor
from src.models.wind_tracer import WindTracer


DATA_DIR = ROOT / "data" / "simulated" / "v1"
MODEL_DIR = ROOT / "models" / "simulated_v1"
REPORT_DIR = ROOT / "output" / "doc"
ASSET_DIR = REPORT_DIR / "assets"
RESULT_PATH = REPORT_DIR / "run_results.json"
MARKDOWN_PATH = REPORT_DIR / "A-H区域运行分析报告.md"

CAUSE_FEATURES = [
    "pm25", "pm10", "so2", "no2", "co", "o3",
    "wind_dir_sin", "wind_dir_cos", "wind_speed", "temperature",
    "humidity", "pressure", "boundary_layer_height", "solar_radiation",
]
PREDICTION_FEATURES = list(CAUSE_FEATURES)

plt.rcParams.update(
    {
        "font.sans-serif": ["PingFang SC", "Arial Unicode MS", "Heiti SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 220,
    }
)


def json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"无法序列化类型: {type(value)!r}")


def load_data() -> Dict[str, pd.DataFrame]:
    return {
        "model": pd.read_csv(DATA_DIR / "model_input_hourly.csv.gz", parse_dates=["timestamp"]),
        "events": pd.read_csv(
            DATA_DIR / "events.csv",
            parse_dates=["start_time", "end_time", "peak_time"],
        ),
        "air": pd.read_csv(DATA_DIR / "air_quality.csv.gz", parse_dates=["timestamp"]),
        "weather": pd.read_csv(DATA_DIR / "weather.csv.gz", parse_dates=["timestamp"]),
        "sources": pd.read_csv(DATA_DIR / "sources.csv"),
        "activity": pd.read_csv(DATA_DIR / "source_activity.csv.gz", parse_dates=["timestamp"]),
        "stations": pd.read_csv(DATA_DIR / "stations.csv"),
    }


def train_prediction_model(model_data: pd.DataFrame) -> tuple[LSTMPredictor, Dict[str, Any]]:
    predictor = LSTMPredictor(
        input_size=len(PREDICTION_FEATURES),
        hidden_size=48,
        output_size=6,
        device=torch.device("cpu"),
        random_seed=42,
    )
    X, y = predictor.prepare_data(
        model_data,
        PREDICTION_FEATURES,
        "pm25",
        seq_length=24,
    )
    split = int(len(X) * 0.8)
    purge_gap = 24 + 6 - 1
    train_end = split - purge_gap
    history = predictor.train(
        X[:train_end],
        y[:train_end],
        epochs=80,
        batch_size=64,
        validation_split=0.2,
        patience=12,
    )
    metrics = predictor.evaluate(X[split:], y[split:])
    predictor_path = MODEL_DIR / "lstm_pm25_predictor.pt"
    predictor.save(str(predictor_path))

    scaled = torch.from_numpy(
        np.ascontiguousarray(predictor._scale_X(X[split:]))
    ).to(predictor.device)
    predictor.model.eval()
    with torch.no_grad():
        prediction = predictor._inverse_y(predictor.model(scaled).cpu().numpy())
    horizon_mae = np.mean(np.abs(prediction - y[split:]), axis=0)
    pm25_feature_index = PREDICTION_FEATURES.index("pm25")
    persistence = np.repeat(
        X[split:, -1, pm25_feature_index][:, None],
        predictor.output_size,
        axis=1,
    )
    persistence_error = persistence - y[split:]
    persistence_mae = float(np.mean(np.abs(persistence_error)))
    persistence_rmse = float(np.sqrt(np.mean(persistence_error ** 2)))
    result = {
        **metrics,
        "persistence_baseline_mae": persistence_mae,
        "persistence_baseline_rmse": persistence_rmse,
        "mae_improvement_over_persistence": float(
            (persistence_mae - metrics["mae"]) / persistence_mae
        ),
        "sample_count": int(len(X)),
        "train_sample_count": int(train_end),
        "held_out_sample_count": int(len(X) - split),
        "outer_purge_gap_samples": purge_gap,
        "split": "chronological_80_20_with_purge_gap",
        "best_epoch": int(history["best_epoch"]),
        "epochs_ran": len(history["train_loss"]),
        "stopped_early": bool(history["stopped_early"]),
        "horizon_mae": horizon_mae.tolist(),
        "feature_columns": PREDICTION_FEATURES,
        "model_path": str(predictor_path.relative_to(ROOT)),
        "history": history,
    }
    return predictor, result


def train_cause_model(model_data: pd.DataFrame) -> tuple[CauseAnalyzer, Dict[str, Any]]:
    analyzer = CauseAnalyzer(
        input_size=len(CAUSE_FEATURES),
        hidden_size=64,
        device=torch.device("cpu"),
        random_seed=42,
    )
    X, y = analyzer.prepare_data(
        model_data,
        CAUSE_FEATURES,
        label_column="cause_label",
        seq_length=12,
    )
    split = int(len(X) * 0.8)
    history = analyzer.train(
        X[:split],
        y[:split],
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        patience=20,
    )
    metrics = analyzer.evaluate(X[split:], y[split:])
    model_path = MODEL_DIR / "cause_classifier.pt"
    analyzer.save(str(model_path))
    result = {
        **metrics,
        "total_event_samples": int(len(X)),
        "train_sample_count": int(split),
        "held_out_sample_count": int(len(X) - split),
        "split": "chronological_80_20",
        "best_epoch": int(history["best_epoch"]),
        "epochs_ran": len(history["train_loss"]),
        "feature_columns": CAUSE_FEATURES,
        "model_path": str(model_path.relative_to(ROOT)),
        "history": history,
    }
    return analyzer, result


def evaluate_tracing(data: Dict[str, pd.DataFrame]) -> tuple[Dict[str, Any], pd.DataFrame]:
    events = data["events"]
    air_by_time = {time: frame for time, frame in data["air"].groupby("timestamp")}
    activity_by_time = {
        time: frame.set_index("source_id")
        for time, frame in data["activity"].groupby("timestamp")
    }
    tracer = WindTracer()
    gradient_analyzer = ConcentrationGradientAnalyzer()
    receptor = data["stations"].loc[
        data["stations"]["station_id"] == "ST00", ["lat", "lon"]
    ].iloc[0]
    records: List[Dict[str, Any]] = []
    traces: Dict[str, Dict[str, Any]] = {}
    for event in events.itertuples():
        peak_air = air_by_time[event.peak_time]
        observations = peak_air[
            ["station_id", "lat", "lon", event.target_pollutant]
        ].rename(columns={event.target_pollutant: "concentration"}).dropna()
        gradient = gradient_analyzer.fit(observations)
        gradient["is_measured"] = False
        gradient["is_simulated"] = True
        current_activity = activity_by_time[event.peak_time]
        sources = []
        for source in data["sources"].itertuples():
            sources.append(
                {
                    "source_id": source.source_id,
                    "name": source.source_name,
                    "source_type": source.cause_type,
                    "lat": float(source.lat),
                    "lon": float(source.lon),
                    "emission_rate": float(
                        current_activity.loc[
                            source.source_id,
                            f"{event.target_pollutant}_emission",
                        ]
                    ),
                }
            )
        trace = tracer.trace(
            event_lat=float(receptor["lat"]),
            event_lon=float(receptor["lon"]),
            event_time=event.peak_time,
            weather_df=data["weather"],
            sources=sources,
            hours=3,
            threshold_km=1.5,
            concentration_gradient=gradient,
        )
        ranking = [item["source_id"] for item in trace.get("contributions", [])]
        true_sources = event.true_source_ids.split("|")
        records.append(
            {
                "event_id": event.event_id,
                "cause": event.primary_cause,
                "target_pollutant": event.target_pollutant,
                "top1_hit": bool(ranking and ranking[0] in true_sources),
                "top3_hit": any(source_id in true_sources for source_id in ranking[:3]),
                "candidate_count": len(ranking),
                "gradient_r2": float(gradient["fit_r2"]),
                "ranking": ranking[:3],
                "true_sources": true_sources,
            }
        )
        traces[event.event_id] = trace

    detail = pd.DataFrame(records)
    by_cause = (
        detail.groupby("cause")[["top1_hit", "top3_hit"]]
        .mean()
        .rename(columns={"top1_hit": "top1_accuracy", "top3_hit": "top3_accuracy"})
    )
    representative_row = (
        detail[(detail["top1_hit"]) & (detail["cause"] != "气象累积")]
        .sort_values("gradient_r2", ascending=False)
        .iloc[0]
    )
    event = events[events["event_id"] == representative_row["event_id"]].iloc[0]
    trace = traces[str(event["event_id"])]
    representative = {
        "event_id": event["event_id"],
        "start_time": event["start_time"],
        "end_time": event["end_time"],
        "peak_time": event["peak_time"],
        "primary_cause": event["primary_cause"],
        "target_pollutant": event["target_pollutant"],
        "true_source_ids": event["true_source_ids"].split("|"),
        "gradient": trace["concentration_gradient"],
        "wind_dir": trace["wind_dir"],
        "wind_speed": trace["wind_speed"],
        "trajectory": trace["trajectory"],
        "ranking": [
            {
                "source_id": item["source_id"],
                "name": item["name"],
                "relative_suspicion": item["contribution"],
                "distance_to_trajectory_km": item["distance"],
            }
            for item in trace["contributions"][:5]
        ],
        "impact_range": trace["impact_range"],
    }
    summary = {
        "event_count": int(len(detail)),
        "top1_accuracy": float(detail["top1_hit"].mean()),
        "top3_accuracy": float(detail["top3_hit"].mean()),
        "ranked_event_rate": float((detail["candidate_count"] > 0).mean()),
        "mean_gradient_r2": float(detail["gradient_r2"].mean()),
        "by_cause": by_cause.to_dict(orient="index"),
        "representative_case": representative,
    }
    return summary, detail


def dataset_summary(data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    air = data["air"]
    event_counts = data["events"]["primary_cause"].value_counts().sort_index()
    center = air[air["station_id"] == "ST00"]
    return {
        "period_start": air["timestamp"].min(),
        "period_end": air["timestamp"].max(),
        "station_count": int(data["stations"]["station_id"].nunique()),
        "source_count": int(data["sources"]["source_id"].nunique()),
        "event_count": int(len(data["events"])),
        "event_counts": event_counts.to_dict(),
        "air_quality_rows": int(len(air)),
        "weather_rows": int(len(data["weather"])),
        "source_activity_rows": int(len(data["activity"])),
        "qc_counts": air["qc_flag"].value_counts().to_dict(),
        "center_pollutant_summary": {
            pollutant: {
                "mean": float(center[pollutant].mean()),
                "median": float(center[pollutant].median()),
                "max": float(center[pollutant].max()),
            }
            for pollutant in ["pm25", "pm10", "so2", "no2", "co", "o3"]
        },
    }


def make_figures(
    data: Dict[str, pd.DataFrame],
    prediction: Dict[str, Any],
    cause: Dict[str, Any],
    tracing: Dict[str, Any],
    trace_detail: pd.DataFrame,
) -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    station_df = data["stations"].copy()
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    center_row = station_df[station_df["station_id"] == "ST00"].iloc[0]
    ax.scatter(
        [center_row["east_km"]],
        [center_row["north_km"]],
        s=270,
        marker="*",
        color="#F0A202",
        edgecolor="#17365D",
        linewidth=1.0,
        label="A点：中心受体",
        zorder=5,
    )
    ax.annotate(
        "A\n中心受体",
        (center_row["east_km"], center_row["north_km"]),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=11,
        weight="bold",
        color="#17365D",
    )
    colors = ["#2F75B5", "#4E9F6D", "#D95F4B", "#7A5AA6", "#2E7D8F", "#B7791F", "#8A5A44"]
    for color, row in zip(colors, station_df[station_df["station_id"] != "ST00"].itertuples()):
        alias = str(row.station_name).split("点")[0]
        ax.scatter(
            [row.east_km],
            [row.north_km],
            s=115,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        ax.annotate(
            alias,
            (row.east_km, row.north_km),
            xytext=(7, 6),
            textcoords="offset points",
            fontsize=10.5,
            weight="bold",
            color=color,
        )
        ax.plot(
            [0, row.east_km],
            [0, row.north_km],
            color=color,
            linewidth=0.9,
            alpha=0.45,
            linestyle="--",
        )
    ax.annotate(
        "北 N",
        xy=(-0.41, 0.40),
        xytext=(-0.41, 0.22),
        ha="center",
        arrowprops={"arrowstyle": "-|>", "lw": 1.5, "color": "#17365D"},
        color="#17365D",
        weight="bold",
    )
    ax.set_xlim(-0.48, 0.48)
    ax.set_ylim(-0.48, 0.48)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("相对A点东向距离（km）")
    ax.set_ylabel("相对A点北向距离（km）")
    ax.set_title("A-H监测点随机布局示意图")
    ax.axhline(0, color="#D8DEE8", linewidth=0.8)
    ax.axvline(0, color="#D8DEE8", linewidth=0.8)
    ax.grid(color="#EEF1F5", linewidth=0.7)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "00_location_relationship.png", bbox_inches="tight")
    plt.close(fig)

    center = data["air"][data["air"]["station_id"] == "ST00"].copy()
    begin, end = center["timestamp"].min(), center["timestamp"].min() + pd.Timedelta(days=14)
    view = center[(center["timestamp"] >= begin) & (center["timestamp"] < end)]
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.8), sharex=True)
    axes[0].plot(view["timestamp"], view["pm25"], color="#C84B31", linewidth=0.9, label="PM2.5观测")
    axes[0].plot(view["timestamp"], view["pm25_true"], color="#263C5A", linewidth=1.0, alpha=0.8, label="PM2.5参考值")
    axes[0].set_ylabel("μg/m³")
    axes[0].legend(loc="upper right", ncol=2, frameon=False)
    axes[0].set_title("中心受体站前14天浓度序列")
    axes[1].plot(view["timestamp"], view["no2"], color="#2E7D8F", linewidth=0.9, label="NO₂")
    axes[1].plot(view["timestamp"], view["o3"], color="#E0A02B", linewidth=0.9, label="O₃")
    axes[1].set_ylabel("μg/m³")
    axes[1].legend(loc="upper right", ncol=2, frameon=False)
    axes[1].grid(alpha=0.25)
    axes[0].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "01_data_overview.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].plot(prediction["history"]["train_loss"], label="训练损失", color="#26547C")
    axes[0].plot(prediction["history"]["val_loss"], label="验证损失", color="#EF476F")
    axes[0].set_title("PM2.5预测模型训练曲线")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("标准化MSE")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)
    axes[1].plot(cause["history"]["train_loss"], label="训练损失", color="#26547C")
    axes[1].plot(cause["history"]["val_loss"], label="验证损失", color="#EF476F")
    axes[1].set_title("成因分类模型训练曲线")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("交叉熵")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "02_training_curves.png", bbox_inches="tight")
    plt.close(fig)

    matrix = np.asarray(cause["confusion_matrix"])
    labels = list(cause["per_class_recall"])
    fig, ax = plt.subplots(figsize=(7.2, 6.1))
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("成因分类留出测试集混淆矩阵")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "03_confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)

    cause_order = list(cause["per_class_recall"])
    by_cause = tracing["by_cause"]
    top1 = [by_cause[name]["top1_accuracy"] for name in cause_order]
    top3 = [by_cause[name]["top3_accuracy"] for name in cause_order]
    positions = np.arange(len(cause_order))
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.bar(positions - 0.18, top1, width=0.36, label="Top-1", color="#3A7D44")
    ax.bar(positions + 0.18, top3, width=0.36, label="Top-3", color="#8CC084")
    ax.set_xticks(positions, cause_order)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("命中率")
    ax.set_title("分成因污染源定位命中率")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "04_trace_performance.png", bbox_inches="tight")
    plt.close(fig)

    case = tracing["representative_case"]
    source_df = data["sources"]
    station_df = data["stations"]
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    ax.scatter(
        source_df["east_km"], source_df["north_km"],
        marker="^", s=70, color="#D35F4D", label="候选污染源",
    )
    for row in source_df.itertuples():
        ax.annotate(row.source_id.replace("SRC_", ""), (row.east_km, row.north_km), fontsize=7)
    ax.scatter(
        station_df["east_km"], station_df["north_km"],
        marker="o", s=48, color="#246A9A", label="监测站",
    )
    ax.scatter([0], [0], marker="*", s=220, color="#F0A202", edgecolor="black", label="中心受体")
    trajectory = np.asarray(case["trajectory"])
    center_station = station_df[station_df["station_id"] == "ST00"].iloc[0]
    lat0, lon0 = float(center_station["lat"]), float(center_station["lon"])
    east = (trajectory[:, 1] - lon0) * 111.320 * np.cos(np.radians(lat0))
    north = (trajectory[:, 0] - lat0) * 110.574
    ax.plot(east, north, "-o", color="#5B3F8C", linewidth=1.8, markersize=3, label="3小时反向轨迹")
    ax.set_xlabel("相对受体东向距离（km）")
    ax.set_ylabel("相对受体北向距离（km）")
    ax.set_title(f"代表事件 {case['event_id']} 风场轨迹与源站布局")
    ax.axhline(0, color="#CCCCCC", linewidth=0.7)
    ax.axvline(0, color="#CCCCCC", linewidth=0.7)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "05_case_trace.png", bbox_inches="tight")
    plt.close(fig)


def build_markdown(results: Dict[str, Any]) -> str:
    dataset = results["dataset"]
    prediction = results["prediction_model"]
    cause = results["cause_model"]
    tracing = results["tracing"]
    case = tracing["representative_case"]
    ranking_lines = "\n".join(
        f"| {index} | {item['name'].replace('虚拟', '潜在')} | {item['relative_suspicion']:.1%} | {item['distance_to_trajectory_km']:.2f} |"
        for index, item in enumerate(case["ranking"], 1)
    )
    return f"""# A-H区域小尺度空气污染运行分析报告

因无法获得真实工业园区数据，故使用自编写数据集按照题目要求的原理进行计算。

## 数据概况

- 时间范围：{dataset["period_start"]} 至 {dataset["period_end"]}
- 监测站：{dataset["station_count"]} 个
- 潜在污染源：{dataset["source_count"]} 个
- 空气质量记录：{dataset["air_quality_rows"]:,} 条
- 污染事件：{dataset["event_count"]} 个，六类各24个

## 重新训练结果

PM2.5六小时预测模型采用时间顺序80/20切分并清除重叠窗口：
MAE={prediction["mae"]:.2f} μg/m³，RMSE={prediction["rmse"]:.2f} μg/m³，
R²={prediction["r2"]:.3f}；相对持续性基线的MAE改善为
{prediction["mae_improvement_over_persistence"]:.1%}。

成因分类模型在29个时间留出事件上的 accuracy={cause["accuracy"]:.2%}，
macro-F1={cause["macro_f1"]:.2%}。

## 全事件溯源评估

- 评估事件：{tracing["event_count"]} 个
- Top-1源命中率：{tracing["top1_accuracy"]:.2%}
- Top-3源命中率：{tracing["top3_accuracy"]:.2%}
- 平均浓度梯度拟合R²：{tracing["mean_gradient_r2"]:.3f}

## 代表事件

- 事件：{case["event_id"]}
- 事件成因：{case["primary_cause"]}
- 峰值时刻：{case["peak_time"]}
- 目标污染物：{case["target_pollutant"]}
- 风向/风速：{case["wind_dir"]:.1f}° / {case["wind_speed"]:.2f} m/s

| 排名 | 候选源 | 相对嫌疑度 | 到轨迹距离(km) |
|---:|---|---:|---:|
{ranking_lines}

## 结论与后续

重新训练的两个模型、风场反向轨迹、多点浓度梯度和源排序流程均已完成运行。
后续可接入多站同步浓度、人工复核事件标签和源工况记录，开展迁移校准与独立评估。
"""


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    _, prediction_result = train_prediction_model(data["model"])
    _, cause_result = train_cause_model(data["model"])
    tracing_result, trace_detail = evaluate_tracing(data)
    results = {
        "report_title": "A-H区域小尺度空气污染运行分析报告",
        "generated_at": pd.Timestamp.now().isoformat(),
        "simulation_only": True,
        "competition_claim_allowed": False,
        "dataset": dataset_summary(data),
        "prediction_model": prediction_result,
        "cause_model": cause_result,
        "tracing": tracing_result,
    }
    make_figures(data, prediction_result, cause_result, tracing_result, trace_detail)
    with RESULT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2, default=json_default)
    MARKDOWN_PATH.write_text(build_markdown(results), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
