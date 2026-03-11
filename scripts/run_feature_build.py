#!/usr/bin/env python3
"""
Run feature build: load data, compute features, save.
Usage: python scripts/run_feature_build.py --inst BTC-USDT-SWAP --out data_samples/features.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from research.db.loaders import load_orderbook_snapshot
from research.features import compute_spread, compute_relative_spread, compute_l1_imbalance, compute_microprice


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inst", default="BTC-USDT-SWAP")
    parser.add_argument("--out", default="data_samples/features.parquet")
    args = parser.parse_args()
    df = load_orderbook_snapshot(args.inst)
    df["spread"] = compute_spread(df)
    df["rel_spread"] = compute_relative_spread(df)
    df["imbalance_l1"] = compute_l1_imbalance(df)
    df["microprice"] = compute_microprice(df)
    df.to_parquet(args.out, index=False)
    print(f"Saved features to {args.out}")


if __name__ == "__main__":
    main()
