"""Parse OKX HFT target column names (binary hits, MFE/MAE regression)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd


def parse_target_name(target_name: str) -> dict[str, object]:
    info: dict[str, object] = {
        "target_name": target_name,
        "kind": None,
        "direction": None,
        "threshold_ticks": np.nan,
        "adverse_ticks": np.nan,
        "horizon_sec": np.nan,
        "symmetric": pd.NA,
        "excursion_type": None,
    }

    m = re.match(
        r"^(long|short)_max_(favorable|adverse)_excursion_ticks_within_(\d+)s$",
        target_name,
    )
    if m:
        direction, excursion_type, horizon = m.groups()
        info.update(
            {
                "kind": "regression",
                "direction": direction,
                "horizon_sec": int(horizon),
                "excursion_type": excursion_type,
            }
        )
        return info

    m = re.match(
        r"^(long|short)_hit_(up|down)_(\d+)t_before_(down|up)_(\d+)t_within_(\d+)s$",
        target_name,
    )
    if m:
        direction, _, threshold, _, adverse, horizon = m.groups()
        info.update(
            {
                "kind": "binary",
                "direction": direction,
                "threshold_ticks": int(threshold),
                "adverse_ticks": int(adverse),
                "horizon_sec": int(horizon),
                "symmetric": int(threshold) == int(adverse),
            }
        )
        return info

    m = re.match(
        r"^(long|short)_hit_(up|down)_(\d+)t_before_(down|up)_within_(\d+)s$",
        target_name,
    )
    if m:
        direction, _, threshold, _, horizon = m.groups()
        info.update(
            {
                "kind": "binary",
                "direction": direction,
                "threshold_ticks": int(threshold),
                "adverse_ticks": int(threshold),
                "horizon_sec": int(horizon),
                "symmetric": True,
            }
        )
        return info

    m = re.match(
        r"^(long|short)_hit_(up|down)_(\d+)t_within_(\d+)s$",
        target_name,
    )
    if m:
        direction, _, threshold, horizon = m.groups()
        info.update(
            {
                "kind": "binary",
                "direction": direction,
                "threshold_ticks": int(threshold),
                "adverse_ticks": np.nan,
                "horizon_sec": int(horizon),
                "symmetric": pd.NA,
            }
        )
    return info


def binary_target_favorable_ticks(target_name: str) -> int | None:
    """Favorable move in ticks for binary hits; None if not a parsed binary."""
    info = parse_target_name(target_name)
    if info.get("kind") != "binary":
        return None
    t = info["threshold_ticks"]
    if isinstance(t, (int, np.integer)):
        return int(t)
    if isinstance(t, float) and np.isfinite(t):
        return int(t)
    return None


def binary_target_is_modeling_eligible(
    target_name: str, *, min_favorable_ticks: int = 260
) -> bool:
    """True if favorable threshold >= ``min_favorable_ticks`` (default 260)."""
    ft = binary_target_favorable_ticks(target_name)
    if ft is None:
        return False
    return ft >= min_favorable_ticks
