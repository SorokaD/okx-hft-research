#!/usr/bin/env python3
"""
Export orderbook sample from TimescaleDB to CSV/Parquet.
Usage: python scripts/export_orderbook_sample.py --inst BTC-USDT-SWAP --out data_samples/orderbook_sample.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from research.db.loaders import load_orderbook_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inst", default="BTC-USDT-SWAP", help="Instrument ID")
    parser.add_argument("--ts-start", default=None, help="Start time (ISO)")
    parser.add_argument("--ts-end", default=None, help="End time (ISO)")
    parser.add_argument("--out", default="data_samples/orderbook_sample.parquet", help="Output path")
    args = parser.parse_args()
    df = load_orderbook_snapshot(args.inst, args.ts_start, args.ts_end)
    df.to_parquet(args.out, index=False)
    print(f"Exported {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
