#!/usr/bin/env python3
"""
Run signal evaluation: load data, build signal and label, compute metrics.
Usage: python scripts/run_signal_eval.py --inst BTC-USDT-SWAP
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from research.db.loaders import load_orderbook_snapshot
from research.labels import build_future_mid_move_label
from research.signals import build_imbalance_signal
from research.backtests import hit_rate, precision_long, precision_short


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inst", default="BTC-USDT-SWAP")
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()
    df = load_orderbook_snapshot(args.inst)
    if "mid_px" not in df.columns:
        df["mid_px"] = (df["bid_px_01"] + df["ask_px_01"]) / 2
    label = build_future_mid_move_label(df, horizon_rows=10)
    signal = build_imbalance_signal(df, threshold=args.threshold)
    hr = hit_rate(label, signal)
    pl = precision_long(label, signal)
    ps = precision_short(label, signal)
    print(f"Hit rate: {hr:.4f}, Precision long: {pl:.4f}, Precision short: {ps:.4f}")


if __name__ == "__main__":
    main()
