from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.target_building.build_targets_dataset import build_targets_dataset
from research.utils.tick_size import infer_tick_size

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("build_targets_260_550_custom")

INPUT_BASE = ROOT / "data" / "reconstructed" / "BTC-USDT-SWAP" / "grid100ms"
OUT = ROOT / "outputs" / "targets" / "btc_usdt_swap_targets_dataset_260_550_h30_300.parquet"

DATE_START = pd.Timestamp("2026-03-26", tz="UTC")
DATE_END_EXCL = pd.Timestamp("2026-04-03", tz="UTC")  # inclusive to 2026-04-02

HORIZONS = [30, 60, 120, 150, 180, 210, 240, 270, 300]
THRESHOLDS = [260, 300, 350, 400, 450, 500, 550]
ADVERSE_05 = [int(t * 0.5) for t in THRESHOLDS]
ADVERSE_075 = [int(t * 0.75) for t in THRESHOLDS]


def _load_period() -> pd.DataFrame:
    files = []
    for d in pd.date_range(
        DATE_START, DATE_END_EXCL - pd.Timedelta(days=1), freq="D", tz="UTC"
    ):
        day_dir = INPUT_BASE / d.strftime("%Y-%m-%d")
        files.extend(sorted(day_dir.glob("*.parquet")))
    if not files:
        raise SystemExit(
            f"No parquet shards found under {INPUT_BASE} for requested period"
        )
    log.info("Loading %d parquet shards", len(files))
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
    df = df[
        (df["ts_event"] >= DATE_START) & (df["ts_event"] < DATE_END_EXCL)
    ].sort_values("ts_event").reset_index(drop=True)
    log.info("Loaded rows in period: %d", len(df))
    return df


def main() -> None:
    df = _load_period()
    tick_size = infer_tick_size(df, price_col="mid_px").tick_size
    log.info("tick_size=%s", tick_size)

    base = build_targets_dataset(
        df,
        tick_size=float(tick_size),
        horizons_sec=HORIZONS,
        thresholds_ticks=THRESHOLDS,
        adverse_ticks=ADVERSE_05,
        sampling_step_rows=1,
        drop_last_incomplete=True,
    )
    alt = build_targets_dataset(
        df,
        tick_size=float(tick_size),
        horizons_sec=HORIZONS,
        thresholds_ticks=THRESHOLDS,
        adverse_ticks=ADVERSE_075,
        sampling_step_rows=1,
        drop_last_incomplete=True,
    )

    add_cols = [
        c
        for c in alt.columns
        if ("_before_down_" in c or "_before_up_" in c) and c not in base.columns
    ]
    out = pd.concat([base, alt[add_cols]], axis=1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    log.info("Saved %s shape=%s", OUT, out.shape)


if __name__ == "__main__":
    main()
