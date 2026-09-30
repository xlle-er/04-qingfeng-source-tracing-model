#!/usr/bin/env python3
"""生成可复现的小尺度空气污染模拟数据集。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.synthetic_dataset import SyntheticAirDatasetBuilder, SyntheticDatasetConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/simulated/v1", help="输出目录")
    parser.add_argument("--start", default="2026-01-01 00:00:00", help="起始本地时间")
    parser.add_argument("--days", type=int, default=90, help="模拟天数")
    parser.add_argument("--frequency-minutes", type=int, default=10, help="空气质量采样间隔")
    parser.add_argument("--events", type=int, default=144, help="污染事件数量")
    parser.add_argument("--seed", type=int, default=20260726, help="随机种子")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = SyntheticDatasetConfig(
        start=args.start,
        days=args.days,
        frequency_minutes=args.frequency_minutes,
        event_count=args.events,
        seed=args.seed,
        output_dir=args.output,
    )
    metadata = SyntheticAirDatasetBuilder(config).write()
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

