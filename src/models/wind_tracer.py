"""小尺度风场反向轨迹与源筛选。

气象风向表示“风从哪个方向来”。因此从受体点回溯时，轨迹应沿
风向本身前进，而不是再加180度。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class WindTracer:
    def __init__(
        self,
        distance_decay_km: float = 1.5,
        transport_decay_km: float = 20.0,
        near_field_km: float = 0.3,
    ):
        self.earth_radius = 6371.0
        self.distance_decay_km = distance_decay_km
        self.transport_decay_km = transport_decay_km
        self.near_field_km = near_field_km

    def _destination(self, lat: float, lon: float, bearing: float, distance_km: float) -> Tuple[float, float]:
        """在球面上按方位角与距离求终点。"""
        angular = distance_km / self.earth_radius
        lat1 = math.radians(lat)
        lon1 = math.radians(lon)
        theta = math.radians(bearing % 360)
        lat2 = math.asin(
            math.sin(lat1) * math.cos(angular)
            + math.cos(lat1) * math.sin(angular) * math.cos(theta)
        )
        lon2 = lon1 + math.atan2(
            math.sin(theta) * math.sin(angular) * math.cos(lat1),
            math.cos(angular) - math.sin(lat1) * math.sin(lat2),
        )
        return math.degrees(lat2), math.degrees(lon2)

    def calculate_backward_trajectory(
        self,
        lat: float,
        lon: float,
        wind_dir: float,
        wind_speed: float,
        hours: int,
    ) -> List[Tuple[float, float]]:
        if hours < 0:
            raise ValueError("hours 不能为负数")
        if wind_speed < 0:
            raise ValueError("wind_speed 不能为负数")
        trajectory = [(float(lat), float(lon))]
        current = trajectory[0]
        for _ in range(hours):
            # m/s * 3.6 = km/h；回溯方向就是风的来向。
            current = self._destination(current[0], current[1], wind_dir, wind_speed * 3.6)
            trajectory.append(current)
        return trajectory

    def calculate_time_varying_trajectory(
        self,
        lat: float,
        lon: float,
        event_time: pd.Timestamp,
        weather_df: pd.DataFrame,
        hours: int,
    ) -> List[Tuple[float, float]]:
        """使用每个回溯时次的风场，而非将事件时风场恒定外推。"""
        required = {"timestamp", "wind_dir", "wind_speed"}
        if not required.issubset(weather_df.columns):
            raise KeyError(f"气象数据缺少列: {sorted(required - set(weather_df.columns))}")
        weather = weather_df[list(required)].copy()
        weather["timestamp"] = pd.to_datetime(weather["timestamp"])
        weather = weather.dropna().sort_values("timestamp")
        if weather.empty:
            raise ValueError("气象数据为空")

        trajectory = [(float(lat), float(lon))]
        current = trajectory[0]
        event_time = pd.Timestamp(event_time)
        for step in range(hours):
            target_time = event_time - pd.Timedelta(hours=step)
            delta = (weather["timestamp"] - target_time).abs()
            idx = delta.idxmin()
            if delta.loc[idx] > pd.Timedelta("90min"):
                raise ValueError(f"{target_time} 附近无可用风场")
            row = weather.loc[idx]
            speed = max(float(row["wind_speed"]), 0.0)
            current = self._destination(current[0], current[1], float(row["wind_dir"]), speed * 3.6)
            trajectory.append(current)
        return trajectory

    def match_sources(
        self,
        trajectory: List[Tuple[float, float]],
        sources: List[Dict],
        threshold_km: float = 5.0,
    ) -> List[Dict]:
        if threshold_km <= 0:
            raise ValueError("threshold_km 必须大于0")
        matches = []
        for source in sources:
            if not {"name", "lat", "lon"}.issubset(source):
                raise KeyError("污染源必须包含 name/lat/lon")
            source_lat, source_lon = float(source["lat"]), float(source["lon"])
            receptor_distance = self.calculate_distance(
                trajectory[0][0], trajectory[0][1], source_lat, source_lon
            )
            if len(trajectory) == 1:
                min_distance, along_hour = receptor_distance, 0.0
            else:
                segment_results = [
                    self._distance_to_segment_km(
                        source_lat, source_lon,
                        trajectory[index], trajectory[index + 1],
                    )
                    for index in range(len(trajectory) - 1)
                ]
                best_index = int(np.argmin([item[0] for item in segment_results]))
                min_distance, fraction = segment_results[best_index]
                along_hour = best_index + fraction

            # 投影仍停在受体点的远场源通常位于下风向/侧风向，不应仅因
            # “离监测点近”被当作上风向命中。只保留300m内的近场源。
            is_upwind = along_hour > 0.01 or receptor_distance <= self.near_field_km
            if min_distance <= threshold_km and is_upwind:
                match = dict(source)
                match.update(
                    distance=min_distance,
                    receptor_distance_km=receptor_distance,
                    trajectory_hour=float(along_hour),
                )
                matches.append(match)
        return matches

    @staticmethod
    def _distance_to_segment_km(
        point_lat: float,
        point_lon: float,
        start: Tuple[float, float],
        end: Tuple[float, float],
    ) -> Tuple[float, float]:
        """在小尺度等距平面近似下计算点到轨迹线段的距离与投影比例。"""
        km_lat = 110.574
        km_lon = 111.320 * math.cos(math.radians(point_lat))
        ax = (start[1] - point_lon) * km_lon
        ay = (start[0] - point_lat) * km_lat
        bx = (end[1] - point_lon) * km_lon
        by = (end[0] - point_lat) * km_lat
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        fraction = 0.0 if length_sq <= 1e-12 else min(max(-(ax * dx + ay * dy) / length_sq, 0.0), 1.0)
        closest_x, closest_y = ax + fraction * dx, ay + fraction * dy
        return math.hypot(closest_x, closest_y), fraction

    @staticmethod
    def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """计算从点1指向点2的初始方位角。"""
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        delta_lon = math.radians(lon2 - lon1)
        east = math.sin(delta_lon) * math.cos(phi2)
        north = (
            math.cos(phi1) * math.sin(phi2)
            - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lon)
        )
        return math.degrees(math.atan2(east, north)) % 360

    def calculate_contribution(
        self,
        matches: List[Dict],
        gradient_bearing: Optional[float] = None,
        gradient_sigma_degrees: float = 45.0,
    ) -> List[Dict]:
        """计算相对嫌疑度，不冒充排放源解析的绝对贡献率。

        权重 = 距离指数衰减 × 排放强度 × 浓度梯度方向一致性。
        ``emission_rate`` 缺失时取1；未提供实测梯度时不施加梯度权重。
        """
        if not matches:
            return []
        results = []
        for match in matches:
            emission = max(float(match.get("emission_rate", 1.0)), 0.0)
            crosswind = math.exp(-float(match["distance"]) / max(self.distance_decay_km, 1e-6))
            receptor_distance = float(match.get("receptor_distance_km", match["distance"]))
            alongwind = math.exp(-receptor_distance / max(self.transport_decay_km, 1e-6))
            transport = crosswind * alongwind
            gradient_alignment = 1.0
            if gradient_bearing is not None:
                source_bearing = float(match.get("source_bearing", gradient_bearing))
                angular_error = abs((source_bearing - gradient_bearing + 180) % 360 - 180)
                gradient_alignment = math.exp(-0.5 * (angular_error / gradient_sigma_degrees) ** 2)
            score = emission * transport * gradient_alignment
            item = dict(match)
            item.update(
                emission_rate=emission,
                transport_score=transport,
                gradient_alignment=gradient_alignment,
                weight=score,
            )
            results.append(item)
        total = sum(item["weight"] for item in results)
        if total <= 0:
            total = float(len(results))
            for item in results:
                item["weight"] = 1.0
        for item in results:
            item["contribution"] = item["weight"] / total
        return sorted(results, key=lambda item: item["contribution"], reverse=True)

    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
        dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        )
        a = min(max(a, 0.0), 1.0)
        return self.earth_radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def trace(
        self,
        event_lat: float,
        event_lon: float,
        event_time: pd.Timestamp,
        weather_df: pd.DataFrame,
        sources: List[Dict],
        hours: int = 6,
        threshold_km: float = 1.5,
        allow_steady_wind_fallback: bool = True,
        concentration_gradient: Optional[Dict] = None,
    ) -> Dict:
        trajectory_mode = "hourly_time_varying_wind"
        try:
            trajectory = self.calculate_time_varying_trajectory(
                event_lat, event_lon, event_time, weather_df, hours
            )
        except (ValueError, KeyError) as exc:
            if not allow_steady_wind_fallback or weather_df.empty:
                return {"error": str(exc), "event_time": event_time}
            weather_fallback = weather_df.copy()
            weather_fallback["timestamp"] = pd.to_datetime(weather_fallback["timestamp"])
            nearest = (weather_fallback["timestamp"] - pd.Timestamp(event_time)).abs().idxmin()
            if abs(weather_fallback.loc[nearest, "timestamp"] - pd.Timestamp(event_time)) > pd.Timedelta("90min"):
                return {"error": str(exc), "event_time": event_time}
            fallback_row = weather_fallback.loc[nearest]
            trajectory = self.calculate_backward_trajectory(
                event_lat,
                event_lon,
                float(fallback_row["wind_dir"]),
                max(float(fallback_row["wind_speed"]), 0.0),
                hours,
            )
            trajectory_mode = "steady_current_wind_screening"

        weather = weather_df.copy()
        weather["timestamp"] = pd.to_datetime(weather["timestamp"])
        nearest = (weather["timestamp"] - pd.Timestamp(event_time)).abs().idxmin()
        row = weather.loc[nearest]
        wind_dir = float(row["wind_dir"])
        wind_speed = float(row["wind_speed"])
        matches = self.match_sources(trajectory, sources, threshold_km=threshold_km)
        for match in matches:
            match["source_bearing"] = self._bearing(
                event_lat, event_lon, float(match["lat"]), float(match["lon"])
            )
        gradient_bearing = None
        if concentration_gradient:
            gradient_bearing = float(concentration_gradient["gradient_bearing"])
        contributions = self.calculate_contribution(matches, gradient_bearing=gradient_bearing)
        top_share = contributions[0]["contribution"] if contributions else 0.0
        confidence = "high" if top_share >= 0.6 else "medium" if top_share >= 0.4 else "low"
        if trajectory_mode == "steady_current_wind_screening":
            confidence = "low"
        impact_distance = wind_speed * 3.6 * hours
        downwind_bearing = (wind_dir + 180) % 360
        sector_polygon = [(float(event_lat), float(event_lon))]
        for bearing in np.linspace(downwind_bearing - 25, downwind_bearing + 25, 9):
            sector_polygon.append(self._destination(event_lat, event_lon, float(bearing), impact_distance))
        sector_polygon.append((float(event_lat), float(event_lon)))
        return {
            "event_location": (event_lat, event_lon),
            "event_time": pd.Timestamp(event_time),
            "wind_dir": wind_dir,
            "wind_speed": wind_speed,
            "trajectory": trajectory,
            "matched_sources": matches,
            "contributions": contributions,
            "source_ranking_confidence": confidence,
            "trajectory_mode": trajectory_mode,
            "concentration_gradient": concentration_gradient,
            "impact_range": {
                "downwind_bearing": downwind_bearing,
                "transport_distance_km": impact_distance,
                "sector_half_angle_degrees": 25.0,
                "sector_polygon": sector_polygon,
            },
            "method_note": (
                ("结果联合使用风场、源清单和多点浓度梯度，仍需人工核查。"
                 if concentration_gradient else
                 "结果仅为风场与源清单约束下的相对嫌疑度；缺少多点浓度梯度。")
                + ("当前仅有即时风场，轨迹采用稳态风近似，置信度已降为 low。"
                   if trajectory_mode == "steady_current_wind_screening" else "")
            ),
        }
