"""机制驱动的小尺度空气污染模拟数据集。

生成结果同时包含带传感器误差的“观测值”和无误差“模拟真值”。浓度并非
相互独立的随机曲线，而是由污染源活动、逐时风场和简化高斯羽流共同决定。
该数据只能用于算法联调和合成基准，不能替代真实案例验证。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


POLLUTANTS = ("pm25", "pm10", "so2", "no2", "co", "o3")
CAUSES = ("工业排放", "机动车尾气", "扬尘", "燃烧源", "二次生成", "气象累积")
TARGET_POLLUTANT = {
    "工业排放": "so2",
    "机动车尾气": "no2",
    "扬尘": "pm10",
    "燃烧源": "pm25",
    "二次生成": "o3",
    "气象累积": "pm25",
}
CONCENTRATION_SCALE = {
    "pm25": 1.8,
    "pm10": 1.6,
    "so2": 0.9,
    "no2": 1.5,
    "co": 0.045,
    "o3": 1.25,
}


@dataclass(frozen=True)
class SyntheticDatasetConfig:
    start: str = "2026-01-01 00:00:00"
    days: int = 90
    frequency_minutes: int = 10
    event_count: int = 144
    seed: int = 20260726
    center_lat: float = 30.000000
    center_lon: float = 120.000000
    timezone: str = "Asia/Shanghai"
    output_dir: str = "data/simulated/v1"

    def validate(self) -> None:
        if self.days < 2:
            raise ValueError("days 至少为2")
        if self.frequency_minutes <= 0 or 60 % self.frequency_minutes != 0:
            raise ValueError("frequency_minutes 必须是60的正约数")
        if self.event_count < len(CAUSES):
            raise ValueError(f"event_count 至少为{len(CAUSES)}")


def _offset_coordinate(lat: float, lon: float, east_km: float, north_km: float) -> Tuple[float, float]:
    return (
        lat + north_km / 110.574,
        lon + east_km / (111.320 * math.cos(math.radians(lat))),
    )


class SyntheticAirDatasetBuilder:
    """创建多站、气象、排放活动、事件标签和源贡献真值。"""

    def __init__(self, config: SyntheticDatasetConfig | None = None):
        self.config = config or SyntheticDatasetConfig()
        self.config.validate()
        self.rng = np.random.default_rng(self.config.seed)
        self.timestamps = pd.date_range(
            self.config.start,
            periods=self.config.days * 24 * 60 // self.config.frequency_minutes,
            freq=f"{self.config.frequency_minutes}min",
        )
        self.stations = self._build_stations()
        self.sources = self._build_sources()
        self.events = self._build_events()
        self.weather = self._build_weather()

    def _build_stations(self) -> pd.DataFrame:
        # A点固定为中心受体，B-H点在局地平面内随机排布。
        # 使用独立固定种子，保证布局可复现且不扰动事件生成随机序列。
        station_rng = np.random.default_rng(self.config.seed + 11)
        layouts = [("ST00", "A点（中心受体）", 0.00, 0.00)]
        offsets: List[tuple[float, float]] = [(0.0, 0.0)]
        for index, label in enumerate("BCDEFGH", start=1):
            for _ in range(1000):
                radius = station_rng.uniform(0.10, 0.35)
                angle = station_rng.uniform(0, 2 * np.pi)
                east = float(radius * np.cos(angle))
                north = float(radius * np.sin(angle))
                if all(np.hypot(east - x, north - y) >= 0.10 for x, y in offsets):
                    offsets.append((east, north))
                    layouts.append((f"ST{index:02d}", f"{label}点", east, north))
                    break
            else:
                raise RuntimeError("无法生成满足最小间距的监测点布局")
        records = []
        for station_id, name, east, north in layouts:
            lat, lon = _offset_coordinate(
                self.config.center_lat, self.config.center_lon, east, north
            )
            records.append(
                {
                    "station_id": station_id,
                    "station_name": name,
                    "lat": lat,
                    "lon": lon,
                    "height_m": 4.0,
                    "east_km": east,
                    "north_km": north,
                    "is_simulated": True,
                }
            )
        return pd.DataFrame(records)

    def _build_sources(self) -> pd.DataFrame:
        # 排放率为合成基准的相对 kg/h 尺度，不代表任何真实单位排放水平。
        definitions = [
            ("SRC_IND_01", "虚拟工业源A", "工业排放", -2.1, 1.4, "point", 35, 2.0, 2.8, 7.5, 5.5, 1.2, 0.0),
            ("SRC_IND_02", "虚拟工业源B", "工业排放", 1.8, 1.6, "point", 25, 1.4, 2.0, 4.2, 3.8, 0.9, 0.0),
            ("SRC_TRA_01", "虚拟道路东段", "机动车尾气", 1.2, 0.1, "line_proxy", 0, 0.8, 1.1, 0.1, 4.5, 3.5, 0.0),
            ("SRC_TRA_02", "虚拟道路西段", "机动车尾气", -1.1, -0.1, "line_proxy", 0, 0.7, 1.0, 0.1, 4.0, 3.1, 0.0),
            ("SRC_DUST_01", "虚拟施工场地A", "扬尘", 1.5, 1.0, "area_proxy", 0, 2.0, 9.0, 0.0, 0.5, 0.2, 0.0),
            ("SRC_DUST_02", "虚拟施工场地B", "扬尘", -1.4, 0.8, "area_proxy", 0, 1.7, 7.5, 0.0, 0.4, 0.2, 0.0),
            ("SRC_COM_01", "虚拟燃烧源A", "燃烧源", -1.2, -1.5, "point", 18, 3.0, 3.8, 3.2, 3.0, 4.2, 0.0),
            ("SRC_COM_02", "虚拟燃烧源B", "燃烧源", 1.0, -1.6, "point", 15, 2.4, 3.1, 2.3, 2.5, 3.5, 0.0),
            ("SRC_SEC_01", "虚拟区域二次生成区", "二次生成", -2.4, 0.1, "area_proxy", 0, 1.8, 2.1, 0.4, 1.6, 0.4, 5.5),
            ("SRC_SEC_02", "虚拟区域二次生成区B", "二次生成", 2.3, -0.4, "area_proxy", 0, 1.5, 1.8, 0.3, 1.4, 0.3, 4.8),
            ("SRC_MET_01", "虚拟区域背景源", "气象累积", -0.2, 2.7, "regional_proxy", 0, 2.2, 3.0, 0.8, 2.0, 1.0, 1.0),
            ("SRC_MET_02", "虚拟区域背景源B", "气象累积", 0.5, -2.7, "regional_proxy", 0, 1.8, 2.5, 0.6, 1.7, 0.8, 0.8),
        ]
        columns = [
            "source_id", "source_name", "cause_type", "east_km", "north_km",
            "geometry_type", "stack_height_m", *[f"base_{p}" for p in POLLUTANTS],
        ]
        result = pd.DataFrame(definitions, columns=columns)
        coordinates = [
            _offset_coordinate(
                self.config.center_lat,
                self.config.center_lon,
                float(row.east_km),
                float(row.north_km),
            )
            for row in result.itertuples()
        ]
        result["lat"] = [item[0] for item in coordinates]
        result["lon"] = [item[1] for item in coordinates]
        result["emission_rate_semantics"] = "synthetic_relative_kg_per_hour"
        result["is_simulated"] = True
        return result

    def _build_events(self) -> pd.DataFrame:
        count = self.config.event_count
        labels = np.resize(np.asarray(CAUSES, dtype=object), count)
        self.rng.shuffle(labels)
        total_hours = self.config.days * 24
        # 预留模型热身窗口，使默认24步序列不会丢失首个事件样本。
        warmup_hours = min(24.0, total_hours * 0.10)
        ending_buffer_hours = min(6.0, total_hours * 0.04)
        available_hours = total_hours - warmup_hours - ending_buffer_hours
        spacing = available_hours / count
        duration_max = max(1.0, min(6.0, spacing * 0.48))
        records: List[Dict[str, object]] = []
        for index, cause in enumerate(labels):
            center_hour = (
                warmup_hours
                + (index + 0.5) * spacing
                + self.rng.uniform(-0.12, 0.12) * spacing
            )
            duration = self.rng.uniform(max(1.0, duration_max * 0.55), duration_max)
            start = pd.Timestamp(self.config.start) + pd.Timedelta(hours=center_hour - duration / 2)
            end = start + pd.Timedelta(hours=duration)
            candidates = self.sources[self.sources["cause_type"] == cause]["source_id"].tolist()
            primary_source = str(self.rng.choice(candidates))
            is_mixed = bool(self.rng.random() < 0.28)
            secondary_source = ""
            secondary_multiplier = 1.0
            if is_mixed:
                other_sources = self.sources[
                    ~self.sources["source_id"].isin(candidates)
                ]["source_id"].tolist()
                secondary_source = str(self.rng.choice(other_sources))
                secondary_multiplier = float(self.rng.uniform(1.8, 3.2))
            records.append(
                {
                    "event_id": f"EVT{index + 1:04d}",
                    "start_time": start.floor(f"{self.config.frequency_minutes}min"),
                    "end_time": end.ceil(f"{self.config.frequency_minutes}min"),
                    "primary_cause": cause,
                    "target_pollutant": TARGET_POLLUTANT[str(cause)],
                    "primary_source_id": primary_source,
                    "primary_multiplier": float(self.rng.uniform(6.0, 12.0)),
                    "secondary_source_id": secondary_source,
                    "secondary_multiplier": secondary_multiplier,
                    "is_mixed_event": is_mixed,
                    "severity": str(self.rng.choice(["轻度", "中度", "重度"], p=[0.35, 0.45, 0.20])),
                    "is_simulated": True,
                }
            )
        return pd.DataFrame(records).sort_values("start_time").reset_index(drop=True)

    def _build_weather(self) -> pd.DataFrame:
        hourly = pd.date_range(self.timestamps[0], periods=self.config.days * 24, freq="h")
        n = len(hourly)
        hour = hourly.hour.to_numpy()
        day_index = np.arange(n) / 24
        direction_steps = self.rng.normal(0, 12, n)
        wind_dir = np.mod(210 + np.cumsum(direction_steps) + 25 * np.sin(2 * np.pi * day_index / 7), 360)
        wind_speed = np.clip(
            2.2 + 1.0 * np.sin(2 * np.pi * (hour - 13) / 24) + self.rng.gamma(1.4, 0.7, n),
            0.2,
            9.0,
        )
        temperature = 10 + 5 * np.sin(2 * np.pi * (hour - 7) / 24) + self.rng.normal(0, 1.2, n)
        humidity = np.clip(72 - 1.8 * (temperature - 10) + self.rng.normal(0, 5, n), 28, 96)
        pressure = 1016 + 5 * np.sin(2 * np.pi * day_index / 8) + self.rng.normal(0, 1.5, n)
        precipitation = np.where(self.rng.random(n) < 0.035, self.rng.gamma(1.2, 1.8, n), 0.0)
        boundary_layer = np.clip(
            350 + 650 * np.maximum(np.sin(np.pi * (hour - 6) / 13), 0) + self.rng.normal(0, 80, n),
            120,
            1500,
        )
        solar = np.clip(650 * np.maximum(np.sin(np.pi * (hour - 6) / 12), 0), 0, None)
        result = pd.DataFrame(
            {
                "timestamp": hourly,
                "wind_dir": wind_dir,
                "wind_speed": wind_speed,
                "temperature": temperature,
                "humidity": humidity,
                "pressure": pressure,
                "precipitation": precipitation,
                "boundary_layer_height": boundary_layer,
                "solar_radiation": solar,
            }
        )
        # 气象累积事件强制形成静稳、高湿、低边界层场景。
        for event in self.events.itertuples():
            mask = (result["timestamp"] >= event.start_time.floor("h")) & (
                result["timestamp"] <= event.end_time.ceil("h")
            )
            if event.primary_cause == "气象累积":
                result.loc[mask, "wind_speed"] = self.rng.uniform(0.15, 0.9, int(mask.sum()))
                result.loc[mask, "humidity"] = self.rng.uniform(82, 96, int(mask.sum()))
                result.loc[mask, "boundary_layer_height"] = self.rng.uniform(120, 260, int(mask.sum()))
            else:
                # 事件定义为“受体站确实受到影响”的高值过程，因此让主源处于
                # 受体上风向。风向仍保留逐小时扰动，不是固定轨迹。
                source = self.sources[
                    self.sources["source_id"] == event.primary_source_id
                ].iloc[0]
                source_bearing = math.degrees(
                    math.atan2(float(source["east_km"]), float(source["north_km"]))
                ) % 360
                result.loc[mask, "wind_dir"] = (
                    source_bearing + self.rng.normal(0, 8, int(mask.sum()))
                ) % 360
                result.loc[mask, "wind_speed"] = self.rng.uniform(1.0, 3.2, int(mask.sum()))
                if event.primary_cause == "二次生成":
                    result.loc[mask, "solar_radiation"] *= 1.25
                    result.loc[mask, "humidity"] = np.minimum(result.loc[mask, "humidity"], 65)
        result["data_source"] = "synthetic-meteorology-v1"
        result["is_simulated"] = True
        return result

    def _weather_at_observation_times(self) -> Dict[str, np.ndarray]:
        hourly_seconds = self.weather["timestamp"].astype("int64").to_numpy() / 1e9
        target_seconds = self.timestamps.astype("int64").to_numpy() / 1e9
        result: Dict[str, np.ndarray] = {}
        unwrapped = np.unwrap(np.radians(self.weather["wind_dir"].to_numpy(float)))
        result["wind_dir"] = np.degrees(
            np.interp(target_seconds, hourly_seconds, unwrapped)
        ) % 360
        for column in [
            "wind_speed", "temperature", "humidity", "pressure", "precipitation",
            "boundary_layer_height", "solar_radiation",
        ]:
            result[column] = np.interp(
                target_seconds,
                hourly_seconds,
                self.weather[column].to_numpy(float),
            )
        return result

    def _event_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        event_id = np.full(len(self.timestamps), "", dtype=object)
        cause = np.full(len(self.timestamps), "", dtype=object)
        severity = np.full(len(self.timestamps), "", dtype=object)
        for event in self.events.itertuples():
            mask = (self.timestamps >= event.start_time) & (self.timestamps <= event.end_time)
            event_id[mask] = event.event_id
            cause[mask] = event.primary_cause
            severity[mask] = event.severity
        return event_id, cause, severity

    def _source_activity(
        self, met: Dict[str, np.ndarray]
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        n_time, n_sources = len(self.timestamps), len(self.sources)
        hour = self.timestamps.hour.to_numpy()
        weekday = self.timestamps.dayofweek.to_numpy()
        emissions = np.zeros((n_sources, n_time, len(POLLUTANTS)), dtype=np.float32)
        frames = []
        for source_index, source in enumerate(self.sources.itertuples()):
            if source.cause_type == "机动车尾气":
                profile = (
                    0.35
                    + 1.25 * np.exp(-((hour - 8) / 1.8) ** 2)
                    + 1.45 * np.exp(-((hour - 18) / 2.2) ** 2)
                ) * np.where(weekday < 5, 1.0, 0.72)
            elif source.cause_type == "扬尘":
                profile = np.where((hour >= 7) & (hour <= 18), 1.0, 0.08)
                profile *= np.where(met["precipitation"] > 0.2, 0.12, 1.0)
            elif source.cause_type == "燃烧源":
                profile = 0.35 + 0.9 * np.exp(-((hour - 20) / 3.2) ** 2)
            elif source.cause_type == "二次生成":
                profile = 0.25 + 0.95 * met["solar_radiation"] / 650
            elif source.cause_type == "气象累积":
                profile = 0.55 + 0.55 / (met["wind_speed"] + 0.5)
            else:
                profile = 0.75 + 0.08 * np.sin(2 * np.pi * hour / 24)
            profile = np.asarray(profile, dtype=float)
            multiplier = np.ones(n_time, dtype=float)
            for event in self.events.itertuples():
                mask = (self.timestamps >= event.start_time) & (self.timestamps <= event.end_time)
                if event.primary_source_id == source.source_id:
                    multiplier[mask] *= event.primary_multiplier
                if event.secondary_source_id == source.source_id:
                    multiplier[mask] *= event.secondary_multiplier
            activity = np.clip(profile * multiplier * self.rng.lognormal(0, 0.08, n_time), 0, None)
            for pollutant_index, pollutant in enumerate(POLLUTANTS):
                emissions[source_index, :, pollutant_index] = (
                    activity * float(getattr(source, f"base_{pollutant}"))
                )
            frame = pd.DataFrame(
                {
                    "timestamp": self.timestamps,
                    "source_id": source.source_id,
                    "activity_multiplier": activity,
                    **{
                        f"{pollutant}_emission": emissions[source_index, :, pollutant_index]
                        for pollutant_index, pollutant in enumerate(POLLUTANTS)
                    },
                    "data_source": "synthetic-source-activity-v1",
                    "is_simulated": True,
                }
            )
            frames.append(frame)
        return pd.concat(frames, ignore_index=True), emissions

    def _transport_cube(self, met: Dict[str, np.ndarray]) -> np.ndarray:
        n_sources, n_times, n_stations = len(self.sources), len(self.timestamps), len(self.stations)
        transport = np.zeros((n_sources, n_times, n_stations), dtype=np.float32)
        downwind = np.radians((met["wind_dir"] + 180) % 360)
        unit_east, unit_north = np.sin(downwind), np.cos(downwind)
        mixing = np.sqrt(500 / np.maximum(met["boundary_layer_height"], 100))
        for source_index, source in enumerate(self.sources.itertuples()):
            east = self.stations["east_km"].to_numpy(float) - float(source.east_km)
            north = self.stations["north_km"].to_numpy(float) - float(source.north_km)
            distance = np.hypot(east, north)
            along = unit_east[:, None] * east[None, :] + unit_north[:, None] * north[None, :]
            cross = np.abs(
                unit_east[:, None] * north[None, :] - unit_north[:, None] * east[None, :]
            )
            sigma = 0.16 + 0.24 * np.sqrt(np.maximum(along, 0) + 0.08)
            plume = np.exp(-(cross ** 2) / (2 * sigma ** 2))
            plume *= np.exp(-np.maximum(along, 0) / 9.0)
            plume *= (along >= -0.20) | (distance[None, :] <= 0.30)
            plume /= met["wind_speed"][:, None] + 0.65
            plume *= mixing[:, None]
            if source.geometry_type == "regional_proxy":
                plume = 0.35 + 0.65 * plume
            transport[source_index] = plume.astype(np.float32)
        return transport

    def _build_air_quality(
        self,
        met: Dict[str, np.ndarray],
        emissions: np.ndarray,
        transport: np.ndarray,
    ) -> pd.DataFrame:
        n_times, n_stations = len(self.timestamps), len(self.stations)
        hour = self.timestamps.hour.to_numpy()
        slow = np.sin(2 * np.pi * np.arange(n_times) / (24 * 6 * 7))
        backgrounds = {
            "pm25": 22 + 5 * slow + 5 / (met["wind_speed"] + 0.8),
            "pm10": 38 + 7 * slow + 4 / (met["wind_speed"] + 0.8),
            "so2": 5 + 0.8 * slow,
            "no2": 17 + 5 * np.exp(-((hour - 8) / 2.5) ** 2) + 6 * np.exp(-((hour - 18) / 3) ** 2),
            "co": 0.55 + 0.08 * slow,
            "o3": 25 + 38 * np.maximum(np.sin(np.pi * (hour - 7) / 12), 0),
        }
        event_id, cause, severity = self._event_arrays()
        accumulation = cause == "气象累积"
        backgrounds["pm25"] = backgrounds["pm25"] + accumulation * (16 + 18 / (met["wind_speed"] + 0.5))
        backgrounds["pm10"] = backgrounds["pm10"] + accumulation * (10 + 10 / (met["wind_speed"] + 0.5))
        frames = []
        true_values: Dict[str, np.ndarray] = {}
        for pollutant_index, pollutant in enumerate(POLLUTANTS):
            contribution = np.sum(
                emissions[:, :, pollutant_index, None] * transport,
                axis=0,
            ) * CONCENTRATION_SCALE[pollutant]
            station_background_factor = 1 + self.rng.normal(0, 0.025, n_stations)
            truth = backgrounds[pollutant][:, None] * station_background_factor[None, :] + contribution
            if pollutant == "o3":
                truth += (cause == "二次生成")[:, None] * (
                    12 + 18 * met["solar_radiation"][:, None] / 650
                )
            true_values[pollutant] = np.clip(truth, 0, None)

        missing_mask = self.rng.random((n_times, n_stations)) < 0.004
        spike_mask = (~missing_mask) & (self.rng.random((n_times, n_stations)) < 0.0015)
        drift_mask = np.zeros((n_times, n_stations), dtype=bool)
        for station_index in [2, 6]:
            begin = int(n_times * (0.42 + 0.12 * station_index / n_stations))
            drift_mask[begin : min(begin + n_times // 30, n_times), station_index] = True

        for station_index, station in enumerate(self.stations.itertuples()):
            frame = pd.DataFrame(
                {
                    "timestamp": self.timestamps,
                    "station_id": station.station_id,
                    "lat": station.lat,
                    "lon": station.lon,
                    "active_event_id": event_id,
                    "cause_label": cause,
                    "event_severity": severity,
                }
            )
            for pollutant in POLLUTANTS:
                truth = true_values[pollutant][:, station_index]
                relative_noise = 0.035 if pollutant != "co" else 0.02
                observed = truth + self.rng.normal(0, np.maximum(truth * relative_noise, 0.05))
                if pollutant in {"pm25", "pm10", "no2"}:
                    observed[spike_mask[:, station_index]] += self.rng.uniform(
                        45, 110, int(spike_mask[:, station_index].sum())
                    )
                observed[drift_mask[:, station_index]] *= 1.12
                observed[missing_mask[:, station_index]] = np.nan
                frame[f"{pollutant}_true"] = truth
                frame[pollutant] = np.clip(observed, 0, None)
            frame["qc_flag"] = np.select(
                [
                    missing_mask[:, station_index],
                    spike_mask[:, station_index],
                    drift_mask[:, station_index],
                ],
                ["missing", "sensor_spike", "sensor_drift"],
                default="valid",
            )
            frame["data_source"] = "synthetic-air-quality-v1"
            frame["is_simulated"] = True
            frames.append(frame)
        return pd.concat(frames, ignore_index=True).sort_values(
            ["timestamp", "station_id"]
        ).reset_index(drop=True)

    def _update_event_peaks(self, air_quality: pd.DataFrame) -> pd.DataFrame:
        center = air_quality[air_quality["station_id"] == "ST00"].copy()
        peaks = []
        for event in self.events.itertuples():
            subset = center[
                (center["timestamp"] >= event.start_time)
                & (center["timestamp"] <= event.end_time)
            ]
            target = f"{event.target_pollutant}_true"
            peaks.append(subset.loc[subset[target].idxmax(), "timestamp"])
        result = self.events.copy()
        result["peak_time"] = peaks
        result["true_source_ids"] = result.apply(
            lambda row: "|".join(
                item for item in [row["primary_source_id"], row["secondary_source_id"]] if item
            ),
            axis=1,
        )
        return result

    def _build_contribution_truth(
        self,
        events: pd.DataFrame,
        emissions: np.ndarray,
        transport: np.ndarray,
    ) -> pd.DataFrame:
        time_lookup = {timestamp: index for index, timestamp in enumerate(self.timestamps)}
        records = []
        for event in events.itertuples():
            time_index = time_lookup[pd.Timestamp(event.peak_time)]
            for station_index, station in enumerate(self.stations.itertuples()):
                for source_index, source in enumerate(self.sources.itertuples()):
                    for pollutant_index, pollutant in enumerate(POLLUTANTS):
                        value = (
                            float(emissions[source_index, time_index, pollutant_index])
                            * float(transport[source_index, time_index, station_index])
                            * CONCENTRATION_SCALE[pollutant]
                        )
                        if value <= 1e-8:
                            continue
                        records.append(
                            {
                                "event_id": event.event_id,
                                "peak_time": event.peak_time,
                                "station_id": station.station_id,
                                "source_id": source.source_id,
                                "pollutant": pollutant,
                                "contribution_concentration": value,
                                "is_true_event_source": source.source_id in event.true_source_ids.split("|"),
                                "is_simulated": True,
                            }
                        )
        result = pd.DataFrame(records)
        totals = result.groupby(
            ["event_id", "station_id", "pollutant"]
        )["contribution_concentration"].transform("sum")
        result["relative_source_share"] = np.where(
            totals > 0, result["contribution_concentration"] / totals, 0.0
        )
        return result

    def _build_model_input(self, air_quality: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        center = air_quality[air_quality["station_id"] == "ST00"].set_index("timestamp")
        numeric = list(POLLUTANTS)
        model = center[numeric].resample("h").mean().reset_index()
        model = model.merge(
            self.weather[
                [
                    "timestamp", "wind_dir", "wind_speed", "temperature", "humidity",
                    "pressure", "precipitation", "boundary_layer_height", "solar_radiation",
                ]
            ],
            on="timestamp",
            how="left",
        )
        wind_radians = np.radians(model["wind_dir"])
        model["wind_dir_sin"] = np.sin(wind_radians)
        model["wind_dir_cos"] = np.cos(wind_radians)
        model["cause_label"] = pd.NA
        model["event_id"] = pd.NA
        hour_lookup = {timestamp: index for index, timestamp in enumerate(model["timestamp"])}
        for event in events.itertuples():
            peak_hour = pd.Timestamp(event.peak_time).floor("h")
            if peak_hour in hour_lookup:
                row_index = hour_lookup[peak_hour]
                model.loc[row_index, "cause_label"] = event.primary_cause
                model.loc[row_index, "event_id"] = event.event_id
        model["data_source"] = "synthetic-model-input-v1"
        model["is_simulated"] = True
        return model

    def build(self) -> Dict[str, pd.DataFrame]:
        met = self._weather_at_observation_times()
        source_activity, emissions = self._source_activity(met)
        transport = self._transport_cube(met)
        air_quality = self._build_air_quality(met, emissions, transport)
        events = self._update_event_peaks(air_quality)
        truth = self._build_contribution_truth(events, emissions, transport)
        model_input = self._build_model_input(air_quality, events)
        return {
            "stations": self.stations.copy(),
            "sources": self.sources.copy(),
            "weather": self.weather.copy(),
            "source_activity": source_activity,
            "air_quality": air_quality,
            "events": events,
            "source_contribution_truth": truth,
            "model_input_hourly": model_input,
        }

    def write(self, output_dir: str | Path | None = None) -> Dict[str, object]:
        output = Path(output_dir or self.config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        dataset = self.build()
        compact_files = {
            "stations": "stations.csv",
            "sources": "sources.csv",
            "events": "events.csv",
        }
        large_files = {
            "weather": "weather.csv.gz",
            "source_activity": "source_activity.csv.gz",
            "air_quality": "air_quality.csv.gz",
            "source_contribution_truth": "source_contribution_truth.csv.gz",
            "model_input_hourly": "model_input_hourly.csv.gz",
        }
        files: Dict[str, str] = {}
        for name, filename in compact_files.items():
            dataset[name].to_csv(output / filename, index=False)
            files[name] = filename
        for name, filename in large_files.items():
            dataset[name].to_csv(output / filename, index=False, compression="gzip")
            files[name] = filename

        metadata = {
            "dataset_name": "A-H匿名区域小尺度空气污染机制数据集",
            "version": "1.0.0",
            "generated_at": datetime.now().astimezone().isoformat(),
            "simulation_only": True,
            "competition_claim_allowed": False,
            "warning": "仅用于合成数据算法验证，不可替代真实案例或真实人工标注评估。",
            "config": asdict(self.config),
            "pollutants": {
                "pm25": "μg/m³", "pm10": "μg/m³", "so2": "μg/m³",
                "no2": "μg/m³", "co": "mg/m³", "o3": "μg/m³",
            },
            "cause_classes": list(CAUSES),
            "row_counts": {name: int(len(frame)) for name, frame in dataset.items()},
            "files": files,
            "provenance": {
                "generator": "src.data.synthetic_dataset.SyntheticAirDatasetBuilder",
                "random_seed": self.config.seed,
                "coordinate_reference": "WGS84 synthetic offsets around configured receptor",
                "time_semantics": f"local naive timestamps; timezone={self.config.timezone}",
            },
        }
        with (output / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
        (output / "README.md").write_text(
            self._dataset_readme(metadata),
            encoding="utf-8",
        )
        return metadata

    @staticmethod
    def _dataset_readme(metadata: Dict[str, object]) -> str:
        counts = metadata["row_counts"]
        return f"""# A-H匿名区域小尺度空气污染机制数据集

本数据集是**纯模拟数据**，用于模型训练、管线联调和溯源算法合成基准，
不得作为真实案例、真实监测结果或污染责任认定依据。

## 数据规模

- 空气质量观测：{counts["air_quality"]:,} 行
- 气象记录：{counts["weather"]:,} 行
- 污染源活动：{counts["source_activity"]:,} 行
- 污染事件：{counts["events"]:,} 个
- 源贡献真值：{counts["source_contribution_truth"]:,} 行

## 文件

- `stations.csv`：A-H八个随机排布监测点及相对坐标。
- `sources.csv`：12个虚拟污染源及基础排放谱。
- `weather.csv.gz`：逐小时气象和局地风场。
- `source_activity.csv.gz`：每个源的高分辨率活动与逐污染物排放率。
- `air_quality.csv.gz`：多站观测值、无噪声真值、事件标签和质量标志。
- `events.csv`：事件起止、峰值、主次源和成因真值。
- `source_contribution_truth.csv.gz`：事件峰值时各站各源贡献真值。
- `model_input_hourly.csv.gz`：兼容现有 LSTM 的中心站小时建模表。
- `metadata.json`：参数、单位、随机种子、行数和来源说明。

## 关键字段

`air_quality.csv.gz` 中 `pm25` 等字段表示含噪声观测，
`pm25_true` 等字段表示模拟器内部无噪声真值。`cause_label` 在整个事件期间
填写；`model_input_hourly.csv.gz` 只在事件峰值小时填写标签，使每个事件只
形成一个分类样本。

`qc_flag` 可能为 `valid`、`missing`、`sensor_spike` 或 `sensor_drift`。
所有文件均通过 `is_simulated=true` 明确标记模拟属性。
"""


def generate_dataset(config: SyntheticDatasetConfig | None = None) -> Dict[str, object]:
    """便捷入口：生成并写入配置指定目录。"""
    builder = SyntheticAirDatasetBuilder(config)
    return builder.write()
