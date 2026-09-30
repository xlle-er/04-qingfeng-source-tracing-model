"""多监测点浓度梯度估计。

梯度方向指向浓度增大的方向，只能作为疑似来向约束，不能单独用于
污染责任认定。坐标在小尺度范围内投影为局地东/北向公里坐标。
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd


class ConcentrationGradientAnalyzer:
    """用平面最小二乘拟合 ``浓度 = a*东向 + b*北向 + c``。"""

    REQUIRED_COLUMNS = {"station_id", "lat", "lon", "concentration"}

    def fit(self, observations: pd.DataFrame) -> Dict[str, object]:
        missing = self.REQUIRED_COLUMNS - set(observations.columns)
        if missing:
            raise KeyError(f"浓度梯度数据缺少列: {sorted(missing)}")
        data = observations[list(self.REQUIRED_COLUMNS)].dropna().copy()
        if data["station_id"].nunique() < 3:
            raise ValueError("浓度梯度至少需要3个不同监测点")

        lat0 = float(data["lat"].mean())
        lon0 = float(data["lon"].mean())
        east_km = (data["lon"].to_numpy(float) - lon0) * 111.320 * math.cos(math.radians(lat0))
        north_km = (data["lat"].to_numpy(float) - lat0) * 110.574
        design = np.column_stack([east_km, north_km, np.ones(len(data))])
        if np.linalg.matrix_rank(design) < 3:
            raise ValueError("监测点近似共线，无法估计二维浓度梯度")

        values = data["concentration"].to_numpy(float)
        coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
        predicted = design @ coefficients
        residual = float(np.sum((values - predicted) ** 2))
        total = float(np.sum((values - values.mean()) ** 2))
        r2 = 1.0 - residual / total if total > 0 else 0.0
        east_gradient, north_gradient = map(float, coefficients[:2])
        strength = math.hypot(east_gradient, north_gradient)
        # 方位角定义：0°北、90°东。
        bearing = math.degrees(math.atan2(east_gradient, north_gradient)) % 360
        return {
            "method": "multi_station_spatial_plane",
            "station_count": int(data["station_id"].nunique()),
            "gradient_bearing": bearing,
            "gradient_strength_per_km": strength,
            "fit_r2": r2,
            "is_measured": True,
        }

