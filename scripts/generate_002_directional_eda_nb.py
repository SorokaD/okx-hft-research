"""Generate notebooks/research/002_directional_trading_horizon_30_180.ipynb."""
from __future__ import annotations

import json
from pathlib import Path


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source if source.endswith("\n") else source + "\n",
    }


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source if source.endswith("\n") else source + "\n",
    }


ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "research" / "002_directional_trading_horizon_30_180.ipynb"

cells: list[dict] = []

cells.append(
    md_cell(
        """# 002 — Directional trading horizon EDA (30–180s)

Исследование движений `mid_px` на горизонтах **30–180 с** и порогах **100–600 тиков** на восстановленном order book (BTC-USDT-SWAP, шаг ~100ms).

- Бинарные метки: достижение ±X тиков по **экстремумам** future max/min внутри временного окна (метка **не** включает текущий бар).
- **TP/SL first hit** на **случайной подвыборке** (Numba) — полный скан всех строк с окном до ~1800 шагов слишком тяжёлый.
- Артефакты: `outputs/eda_directional_horizon_30_180/`.
"""
    )
)

cells.append(
    code_cell(
        """# --- Config ---
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", context="notebook")

HORIZONS_SEC = [30, 60, 90, 120, 180]
THRESHOLDS_TICKS = [100, 200, 300, 400, 500, 600]

TP_SL_COMBOS = [
    {"name": "tp200_sl100", "tp_ticks": 200, "sl_ticks": 100},
    {"name": "tp300_sl150", "tp_ticks": 300, "sl_ticks": 150},
    {"name": "tp400_sl200", "tp_ticks": 400, "sl_ticks": 200},
]

SUBSAMPLE_N = 300_000
SUBSAMPLE_SEED = 42
FI_SAMPLE_N = 200_000
FI_SEED = 43
TIME_STATS_N = 80_000
TIME_STATS_SEED = 7

_cwd = Path.cwd().resolve()
_cands = [_cwd, _cwd.parent, _cwd.parent.parent, _cwd.parent.parent.parent]
ROOT = None
for c in _cands:
    if (c / "research").is_dir() and (c / "notebooks").is_dir():
        ROOT = c
        break
if ROOT is None:
    raise RuntimeError("repo root not found")

OUTPUT_DIR = ROOT / "outputs" / "eda_directional_horizon_30_180"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_PATH = ROOT / "data" / "processed" / "modeling" / (
    "btc_usdt_swap_model_dataset_100ms_baseline_20260323_000000__20260325_235959.parquet"
)
INSTRUMENT_LABEL = "BTC-USDT-SWAP"

print("ROOT:", ROOT)
print("PARQUET exists:", PARQUET_PATH.exists())
print("OUTPUT_DIR:", OUTPUT_DIR)
"""
    )
)

cells.append(md_cell("## 1) Загрузка и подготовка"))

cells.append(
    code_cell(
        """import pyarrow.parquet as pq

requested = ["ts_event", "mid_px", "spread_px"]
requested += [f"bid_px_{i:02d}" for i in range(1, 11)]
requested += [f"bid_sz_{i:02d}" for i in range(1, 11)]
requested += [f"ask_px_{i:02d}" for i in range(1, 11)]
requested += [f"ask_sz_{i:02d}" for i in range(1, 11)]

schema_names = pq.read_schema(PARQUET_PATH).names
cols = [c for c in requested if c in schema_names]
missing = [c for c in requested if c not in schema_names]
if "ts_event" not in cols or "mid_px" not in cols:
    raise ValueError("Need ts_event and mid_px in parquet")
print("optional columns missing:", len(missing))
if missing[:8]:
    print("  e.g.", missing[:8])

df = pd.read_parquet(PARQUET_PATH, columns=cols)
df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
df[["mid_px", "spread_px"]] = df[["mid_px", "spread_px"]].apply(pd.to_numeric, errors="coerce")
bid_ask = [c for c in df.columns if c.startswith(("bid_", "ask_"))]
for c in bid_ask:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["ts_event", "mid_px"])
n0 = len(df)
df = df.sort_values("ts_event").drop_duplicates(subset=["ts_event"], keep="last").reset_index(drop=True)
n_dup = n0 - len(df)

mid_na = int(df["mid_px"].isna().sum())
spr_na = int(df["spread_px"].isna().sum()) if "spread_px" in df.columns else 0
ts = pd.Series(df["ts_event"].values)

dt_sec = ts.diff().dt.total_seconds()
df["dt_sec"] = dt_sec.astype("float64")
med_dt = float(dt_sec.median())
p95_dt = float(dt_sec.quantile(0.95))

prep_summary = {
    "instrument": INSTRUMENT_LABEL,
    "parquet": str(PARQUET_PATH),
    "n_rows_after_clean": int(len(df)),
    "n_dup_removed": int(n_dup),
    "mid_px_na": mid_na,
    "spread_px_na": spr_na,
    "t_min": str(df["ts_event"].iloc[0]),
    "t_max": str(df["ts_event"].iloc[-1]),
    "median_dt_sec": float(med_dt),
    "p95_dt_sec": float(p95_dt),
    "frac_dt_gt_0_15s": float((dt_sec > 0.15).mean()),
    "frac_dt_gt_0_5s": float((dt_sec > 0.5).mean()),
}

print(json.dumps(prep_summary, indent=2))
df.head()
"""
    )
)

cells.append(
    code_cell(
        """fig, ax = plt.subplots(figsize=(9, 4))
sns.histplot(dt_sec.clip(0, 2), bins=80, ax=ax, color="steelblue")
ax.set_xlabel("dt_sec (capped at 2s for display)")
ax.set_title("Distribution of spacing between consecutive ts_event")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "hist_dt_sec.png", dpi=160)
plt.show()
"""
    )
)

cells.append(md_cell("## 2) tick_size"))

cells.append(
    code_cell(
        """from research.utils.tick_size import infer_tick_size


def infer_tick_size_from_series(mid: np.ndarray, sample_n: int = 500_000, digits: int = 12) -> float:
    x = np.asarray(mid, dtype="float64")
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    step = max(1, x.size // sample_n)
    xs = np.unique(np.round(x[::step], digits))
    xs.sort()
    d = np.diff(xs)
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        return np.nan
    q = float(np.quantile(d, 0.001))
    return float(np.min(d[d >= max(q, 1e-12)]))


_tick = infer_tick_size_from_series(df["mid_px"].to_numpy())
try:
    _tick2 = float(infer_tick_size(df, price_col="mid_px").tick_size)
except Exception:
    _tick2 = np.nan
if np.isfinite(_tick2) and abs(_tick2 - _tick) / _tick < 0.2:
    tick_size = _tick2
else:
    tick_size = _tick

print("tick_size (auto):", tick_size)
if np.isnan(tick_size):
    raise ValueError("tick_size is NaN")
"""
    )
)

cells.append(md_cell("## 3) Future max/min и бинарные метки"))

cells.append(
    code_cell(
        """# Row windows from median spacing (~time horizon on irregular grid; prefer uniform 100ms grid)
df["mid_px"] = pd.to_numeric(df["mid_px"], errors="coerce")
mid = df["mid_px"].to_numpy(dtype="float64")
ROW_WINDOW = {T: max(1, int(np.round(T / max(float(med_dt), 1e-9)))) for T in HORIZONS_SEC}

for T in HORIZONS_SEC:
    w = ROW_WINDOW[T]
    shifted = df["mid_px"].shift(-1)
    df[f"future_max_mid_{T}s"] = shifted.rolling(w, min_periods=1).max()
    df[f"future_min_mid_{T}s"] = shifted.rolling(w, min_periods=1).min()

for T in HORIZONS_SEC:
    mx = df[f"future_max_mid_{T}s"].to_numpy()
    mn = df[f"future_min_mid_{T}s"].to_numpy()
    df[f"up_move_ticks_{T}s"] = (mx - mid) / tick_size
    df[f"down_move_ticks_{T}s"] = (mid - mn) / tick_size
    for x in THRESHOLDS_TICKS:
        df[f"reached_plus_{x}_by_{T}s"] = (df[f"up_move_ticks_{T}s"] >= x - 1e-9).astype("int8")
        df[f"reached_minus_{x}_by_{T}s"] = (df[f"down_move_ticks_{T}s"] >= x - 1e-9).astype("int8")
"""
    )
)

cells.append(md_cell("### 3b) TP/SL first hit (подвыборка, Numba)"))

cells.append(
    code_cell(
        """rng = np.random.default_rng(SUBSAMPLE_SEED)

sub_n = min(SUBSAMPLE_N, len(df))
sub_idx = rng.choice(len(df), size=sub_n, replace=False)
sub_idx.sort()

mid_arr = df["mid_px"].to_numpy(dtype="float64")
ts_arr = df["ts_event"].to_numpy(dtype="datetime64[ns]")


def horizon_ns(h_sec: float) -> np.int64:
    return np.int64(np.round(float(h_sec) * 1e9))


try:
    from numba import njit

    @njit(cache=True)
    def tp_sl_scan(mid, ts, idx, max_steps, horizon_ns_val, tp_ticks, sl_ticks, tick):
        nout = idx.shape[0]
        outcome = np.empty(nout, dtype=np.int8)
        t_tp = np.empty(nout, dtype=np.float64)
        t_sl = np.empty(nout, dtype=np.float64)
        for k in range(nout):
            i = idx[k]
            mi = mid[i]
            tend = ts[i] + horizon_ns_val
            tp_px = mi + tp_ticks * tick
            sl_px = mi - sl_ticks * tick
            hit_tp = 0
            hit_sl = 0
            t_tp[k] = np.nan
            t_sl[k] = np.nan
            lim = min(mid.shape[0] - 1, i + max_steps)
            for j in range(i + 1, lim + 1):
                if ts[j] > tend:
                    break
                pj = mid[j]
                if pj >= tp_px:
                    t_tp[k] = (ts[j] - ts[i]) / 1e9
                    hit_tp = 1
                    break
                if pj <= sl_px:
                    t_sl[k] = (ts[j] - ts[i]) / 1e9
                    hit_sl = 1
                    break
            if hit_tp:
                outcome[k] = 1
            elif hit_sl:
                outcome[k] = -1
            else:
                outcome[k] = 0
        return outcome, t_tp, t_sl

    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False


def tp_sl_slow(mid, ts, idx, max_steps, horizon_ns_val, tp_ticks, sl_ticks, tick):
    nout = idx.shape[0]
    outcome = np.zeros(nout, dtype=np.int8)
    t_tp = np.full(nout, np.nan)
    t_sl = np.full(nout, np.nan)
    for k in range(nout):
        i = int(idx[k])
        mi = float(mid[i])
        tend = int(ts[i]) + int(horizon_ns_val)
        tp_px = mi + float(tp_ticks) * tick
        sl_px = mi - float(sl_ticks) * tick
        hit_tp = hit_sl = False
        lim = min(mid.shape[0] - 1, i + max_steps)
        for j in range(i + 1, lim + 1):
            if int(ts[j]) > tend:
                break
            pj = float(mid[j])
            if pj >= tp_px:
                t_tp[k] = (int(ts[j]) - int(ts[i])) / 1e9
                hit_tp = True
                break
            if pj <= sl_px:
                t_sl[k] = (int(ts[j]) - int(ts[i])) / 1e9
                hit_sl = True
                break
        if hit_tp:
            outcome[k] = 1
        elif hit_sl:
            outcome[k] = -1
    return outcome, t_tp, t_sl


for c in TP_SL_COMBOS:
    name = c["name"]
    for T in HORIZONS_SEC:
        col_out = f"outcome_{name}_by_{T}s"
        col_ttp = f"time_to_tp_{name}_by_{T}s"
        col_tsl = f"time_to_sl_{name}_by_{T}s"
        df[col_out] = np.nan
        df[col_ttp] = np.nan
        df[col_tsl] = np.nan

for T in HORIZONS_SEC:
    hns = horizon_ns(float(T))
    ms = int(np.clip(int(np.ceil(T / max(med_dt, 1e-6))) + 5, 10, 10_000))
    for c in TP_SL_COMBOS:
        name = c["name"]
        if HAVE_NUMBA:
            oc, ttp, tsl = tp_sl_scan(
                mid_arr,
                ts_arr,
                sub_idx.astype(np.int64),
                ms,
                hns,
                float(c["tp_ticks"]),
                float(c["sl_ticks"]),
                float(tick_size),
            )
        else:
            oc, ttp, tsl = tp_sl_slow(
                mid_arr,
                ts_arr,
                sub_idx.astype(np.int64),
                ms,
                hns,
                float(c["tp_ticks"]),
                float(c["sl_ticks"]),
                float(tick_size),
            )
        df.loc[sub_idx, f"outcome_{name}_by_{T}s"] = oc
        df.loc[sub_idx, f"time_to_tp_{name}_by_{T}s"] = ttp
        df.loc[sub_idx, f"time_to_sl_{name}_by_{T}s"] = tsl

df["target_hit_first_tp300_sl150_60s"] = (df["outcome_tp300_sl150_by_60s"] == 1) & df["outcome_tp300_sl150_by_60s"].notna()
df["stop_hit_first_tp300_sl150_60s"] = (df["outcome_tp300_sl150_by_60s"] == -1) & df["outcome_tp300_sl150_by_60s"].notna()
df["timeout_tp300_sl150_60s"] = (df["outcome_tp300_sl150_by_60s"] == 0) & df["outcome_tp300_sl150_by_60s"].notna()
print("numba:", HAVE_NUMBA)
print(df["outcome_tp300_sl150_by_60s"].value_counts(dropna=True).head())
"""
    )
)

cells.append(md_cell("## 4) Features"))

cells.append(
    code_cell(
        """def col_sum(prefix: str, lo: int, hi: int) -> pd.Series:
    parts = [df[f"{prefix}{i:02d}"] for i in range(lo, hi + 1) if f"{prefix}{i:02d}" in df.columns]
    if not parts:
        return pd.Series(np.nan, index=df.index)
    return pd.concat(parts, axis=1).sum(axis=1)


df["spread_ticks"] = df["spread_px"] / tick_size

mid_idx = df["mid_px"]
df["mid_return_1s"] = mid_idx.pct_change(periods=10)
df["mid_return_5s"] = mid_idx.pct_change(periods=50)
df["mid_return_10s"] = mid_idx.pct_change(periods=100)
df["mid_return_30s"] = mid_idx.pct_change(periods=300)
df["log_mid"] = np.log(mid_idx)

for wname, periods in [("5s", 50), ("10s", 100), ("30s", 300), ("60s", 600)]:
    df[f"rolling_vol_{wname}"] = df["log_mid"].diff().rolling(periods, min_periods=max(5, periods // 10)).std()

for k in (1, 3, 5, 10):
    bs = col_sum("bid_sz_", 1, k)
    zas = col_sum("ask_sz_", 1, k)
    den = bs + zas
    df[f"imbalance_top{k}"] = (bs - zas) / den.replace(0, np.nan)

df["total_bid_sz_1"] = col_sum("bid_sz_", 1, 1)
df["total_ask_sz_1"] = col_sum("ask_sz_", 1, 1)
for k in (5, 10):
    df[f"total_bid_sz_{k}"] = col_sum("bid_sz_", 1, k)
    df[f"total_ask_sz_{k}"] = col_sum("ask_sz_", 1, k)

imb5 = df["imbalance_top5"]
df["imbalance_change_1s"] = imb5.diff(10)
df["imbalance_change_5s"] = imb5.diff(50)
df["spread_change_1s"] = df["spread_px"].diff(10)
df["spread_change_5s"] = df["spread_px"].diff(50)

def slope(y: pd.Series, n: int) -> pd.Series:
    return (y - y.shift(n)) / n


df["mid_slope_5s"] = slope(mid_idx, 50)
df["mid_slope_10s"] = slope(mid_idx, 100)
df["mid_slope_30s"] = slope(mid_idx, 300)


def persistence_seconds(mask: pd.Series, max_seconds: int = 300) -> pd.Series:
    m = mask.fillna(False).to_numpy()
    out = np.zeros(len(m), dtype=np.float64)
    run = 0
    max_steps = int(max_seconds / max(med_dt, 1e-6)) + 2
    dt = float(med_dt)
    for i in range(len(m)):
        if m[i]:
            run += 1
        else:
            run = 0
        out[i] = min(run * dt, max_seconds)
    return pd.Series(out, index=mask.index)


df["persist_imb5_pos_sec"] = persistence_seconds(df["imbalance_top5"] > 0)
df["persist_imb5_neg_sec"] = persistence_seconds(df["imbalance_top5"] < 0)
df["persist_ret1_pos_sec"] = persistence_seconds(df["mid_return_1s"] > 0)
df["persist_ret1_neg_sec"] = persistence_seconds(df["mid_return_1s"] < 0)

b1 = df["bid_sz_01"] if "bid_sz_01" in df.columns else pd.Series(np.nan, index=df.index)
a1 = df["ask_sz_01"] if "ask_sz_01" in df.columns else pd.Series(np.nan, index=df.index)
df["bid_sz_01_drop_pct"] = -(b1.diff()) / b1.shift(1).replace(0, np.nan)
df["ask_sz_01_drop_pct"] = -(a1.diff()) / a1.shift(1).replace(0, np.nan)
"""
    )
)

cells.append(md_cell("## 5) Volatility regime (tercile rolling_vol_30s)"))

cells.append(
    code_cell(
        """rv = df["rolling_vol_30s"]
ranks = rv.rank(pct=True)
df["vol_regime"] = pd.cut(
    ranks,
    bins=[0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
    labels=["low", "medium", "high"],
    include_lowest=True,
)
print(df["vol_regime"].value_counts())

q1, q2 = rv.quantile([1 / 3, 2 / 3])
fig, ax = plt.subplots(figsize=(8, 4))
sns.kdeplot(rv.dropna(), ax=ax, fill=True, color="slategray")
ax.axvline(q1, color="crimson", ls="--", label="rv q(1/3)")
ax.axvline(q2, color="darkorange", ls="--", label="rv q(2/3)")
ax.set_title("rolling_vol_30s distribution")
ax.legend()
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "kde_rolling_vol_30s.png", dpi=160)
plt.show()
"""
    )
)

cells.append(md_cell("## 6) Главный EDA: доли и времена"))


cells.append(
    code_cell(
        """# Доли — по всему df; времена до первого пересечения — на подвыборке стартов (TIME_STATS_N), шаг ~med_dt
mid_arr = df["mid_px"].to_numpy(dtype="float64")
ROW_WINDOW = {T: max(1, int(np.round(T / max(float(med_dt), 1e-9)))) for T in HORIZONS_SEC}
rng_ts = np.random.default_rng(TIME_STATS_SEED)
time_idx = rng_ts.choice(len(df), size=min(TIME_STATS_N, len(df)), replace=False).astype(np.int64)
time_idx.sort()

try:
    from numba import njit

    @njit(cache=True)
    def time_to_first_cross(mid, idx, w, ticks_needed, tick, upward, dt_step):
        out = np.empty(idx.shape[0], dtype=np.float64)
        n = mid.shape[0]
        for k in range(idx.shape[0]):
            i = int(idx[k])
            out[k] = np.nan
            if i + 1 >= n:
                continue
            if upward:
                tgt = mid[i] + ticks_needed * tick
            else:
                tgt = mid[i] - ticks_needed * tick
            lim = i + w
            if lim >= n:
                lim = n - 1
            for j in range(i + 1, lim + 1):
                if upward and mid[j] >= tgt - 1e-12:
                    out[k] = float(j - i) * dt_step
                    break
                if (not upward) and mid[j] <= tgt + 1e-12:
                    out[k] = float(j - i) * dt_step
                    break
        return out

    HAVE_NUMBA_TT = True
except Exception:

    def time_to_first_cross(mid, idx, w, ticks_needed, tick, upward, dt_step):
        out = np.full(idx.shape[0], np.nan)
        n = mid.shape[0]
        for k in range(idx.shape[0]):
            i = int(idx[k])
            if i + 1 >= n:
                continue
            if upward:
                tgt = mid[i] + ticks_needed * tick
            else:
                tgt = mid[i] - ticks_needed * tick
            lim = min(n - 1, i + w)
            for j in range(i + 1, lim + 1):
                if upward and mid[j] >= tgt - 1e-12:
                    out[k] = float(j - i) * dt_step
                    break
                if (not upward) and mid[j] <= tgt + 1e-12:
                    out[k] = float(j - i) * dt_step
                    break
        return out

    HAVE_NUMBA_TT = False

rows = []
for T in HORIZONS_SEC:
    w = ROW_WINDOW[T]
    for x in THRESHOLDS_TICKS:
        cp = f"reached_plus_{x}_by_{T}s"
        cm = f"reached_minus_{x}_by_{T}s"
        reached_p = df[cp].astype(bool)
        reached_m = df[cm].astype(bool)
        dt_step = float(med_dt)
        ttp = time_to_first_cross(mid_arr, time_idx, w, float(x), float(tick_size), True, dt_step)
        ttm = time_to_first_cross(mid_arr, time_idx, w, float(x), float(tick_size), False, dt_step)
        rows.append({
            "horizon_sec": T,
            "threshold_ticks": x,
            "share_reached_plus": float(reached_p.mean()),
            "share_reached_minus": float(reached_m.mean()),
            "median_time_to_plus": float(np.nanmedian(ttp)),
            "median_time_to_minus": float(np.nanmedian(ttm)),
            "p75_time_to_plus": float(np.nanquantile(ttp, 0.75)),
            "p75_time_to_minus": float(np.nanquantile(ttm, 0.75)),
        })

main_eda_df = pd.DataFrame(rows)
main_eda_df.to_parquet(OUTPUT_DIR / "main_eda_summary.parquet", index=False)
print("time-to-cross numba:", HAVE_NUMBA_TT, "| median dt_sec:", med_dt, "| time sample:", len(time_idx))
main_eda_df.head(12)
"""
    )
)

cells.append(
    code_cell(
        """fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, col, title in zip(
    axes,
    ["share_reached_plus", "share_reached_minus"],
    ["P(reach +ticks by horizon)", "P(reach -ticks by horizon)"],
):
    pivot = main_eda_df.pivot(index="threshold_ticks", columns="horizon_sec", values=col)
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax)
    ax.set_title(title)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "heatmap_reach_shares.png", dpi=160)
plt.show()

curve_df = []
for T in HORIZONS_SEC:
    sub = main_eda_df[main_eda_df["horizon_sec"] == T]
    for x in THRESHOLDS_TICKS:
        r = sub[sub["threshold_ticks"] == x].iloc[0]
        curve_df.append({"horizon_sec": T, "threshold_ticks": x, "side": "plus", "p": r["share_reached_plus"]})
        curve_df.append({"horizon_sec": T, "threshold_ticks": x, "side": "minus", "p": r["share_reached_minus"]})
curves = pd.DataFrame(curve_df)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, side, title in zip(axes, ["plus", "minus"], ["Upward reach probability", "Downward reach probability"]):
    sub = curves[curves["side"] == side]
    for x in THRESHOLDS_TICKS:
        ssub = sub[sub["threshold_ticks"] == x].sort_values("horizon_sec")
        ax.plot(ssub["horizon_sec"], ssub["p"], marker="o", label=f"{x} ticks")
    ax.set_title(title)
    ax.set_xlabel("horizon_sec")
    ax.set_ylabel("share reached")
    ax.legend(title="threshold", fontsize=8, ncol=2)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "line_reach_by_horizon.png", dpi=160)
plt.show()
"""
    )
)

cells.append(md_cell("## 7) Regime analysis"))

cells.append(
    code_cell(
        """from IPython.display import display

reg_rows = []
for regime in ["low", "medium", "high"]:
    sub = df[df["vol_regime"] == regime]
    if sub.empty:
        continue
    for x in THRESHOLDS_TICKS:
        cp = f"reached_plus_{x}_by_60s"
        if cp not in sub.columns:
            continue
        reg_rows.append({
            "vol_regime": regime,
            "threshold_ticks": x,
            "p_plus_60s": float(sub[cp].mean()),
            "median_time_plus_60s": float("nan"),
        })
regime_tbl = pd.DataFrame(reg_rows)

if "mid_arr" in dir() and "time_to_first_cross" in dir():
    w60 = max(1, int(np.round(60 / max(float(med_dt), 1e-9))))
    ttp300 = time_to_first_cross(
        mid_arr, time_idx, w60, 300.0, float(tick_size), True, float(med_dt)
    )
    rt = pd.DataFrame(
        {"vol_regime": df["vol_regime"].iloc[time_idx].values, "ttp_plus300_60s": ttp300}
    )
    reg_med = rt.groupby("vol_regime", observed=False)["ttp_plus300_60s"].median().reset_index()
    reg_med.to_parquet(OUTPUT_DIR / "regime_median_ttp_plus300_60s.parquet", index=False)
    display(reg_med)

for i, regime in enumerate(["low", "high"]):
    if regime not in regime_tbl["vol_regime"].values:
        continue

tbl_p = regime_tbl.pivot(index="threshold_ticks", columns="vol_regime", values="p_plus_60s")
if "low" in tbl_p.columns and "high" in tbl_p.columns:
    tbl_p["high_minus_low"] = tbl_p["high"] - tbl_p["low"]
    display(tbl_p)
    tbl_p.to_parquet(OUTPUT_DIR / "regime_plus60_table.parquet")

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(data=regime_tbl, x="threshold_ticks", y="p_plus_60s", hue="vol_regime", ax=ax)
ax.set_title("P(+threshold by 60s) by volatility regime")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "bar_regime_p_plus60.png", dpi=160)
plt.show()
"""
    )
)

cells.append(md_cell("## 8) Feature screening"))

cells.append(
    code_cell(
        """from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif

LABEL_A = "reached_plus_300_by_60s"
LABEL_B = "y_tp300_sl150_60s"


def precision_top10pct(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = y_true.astype(bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    order = np.argsort(-scores)
    k = max(1, int(0.1 * len(y)))
    top = order[:k]
    return float(y[top].mean())


exclude = set()
for c in df.columns:
    if c.startswith("reached_") or c.startswith("future_") or c.startswith("outcome_") or c.startswith("time_to_"):
        exclude.add(c)
    if c.startswith("up_move") or c.startswith("down_move"):
        exclude.add(c)
    if c in ("ts_event", "vol_regime", "target_hit_first_tp300_sl150_60s", "stop_hit_first_tp300_sl150_60s", "timeout_tp300_sl150_60s"):
        exclude.add(c)

fi_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
fi_cols = [c for c in fi_cols if df[c].notna().mean() > 0.5]

rng_fi = np.random.default_rng(FI_SEED)
fi_n = min(FI_SAMPLE_N, len(df))
fi_idx = rng_fi.choice(len(df), size=fi_n, replace=False)

dfa = df.iloc[fi_idx].copy()
ya = dfa[LABEL_A].astype(int).to_numpy()

sub_b = dfa["outcome_tp300_sl150_by_60s"].notna()
dfb = dfa.loc[sub_b].copy()
dfb[LABEL_B] = (dfb["outcome_tp300_sl150_by_60s"] == 1).astype(int)
yb = dfb[LABEL_B].astype(int).to_numpy()

rows_fi = []
for col in fi_cols:
    xa = pd.to_numeric(dfa[col], errors="coerce").to_numpy()
    xb = pd.to_numeric(dfb[col], errors="coerce").to_numpy()
    sa = np.isfinite(xa)
    sb = np.isfinite(xb)
    if sa.sum() < 1000:
        continue
    corr_a = np.corrcoef(xa[sa], ya[sa])[0, 1] if sa.sum() > 2 else np.int64(0)
    mi_a = mutual_info_classif(xa[sa].reshape(-1, 1), ya[sa], random_state=0)[0] if sa.sum() > 2 else np.nan
    try:
        auc_a = roc_auc_score(ya[sa], xa[sa]) if len(np.unique(ya[sa])) > 1 else np.nan
    except ValueError:
        auc_a = np.nan
    p10_a = precision_top10pct(ya[sa], xa[sa])

    sb2 = sb & np.isfinite(xb)
    if sb2.sum() > 1000 and len(np.unique(yb[sb2])) > 1:
        try:
            auc_b = roc_auc_score(yb[sb2], xb[sb2])
        except ValueError:
            auc_b = np.nan
        corr_b = np.corrcoef(xb[sb2], yb[sb2])[0, 1]
        mi_b = mutual_info_classif(xb[sb2].reshape(-1, 1), yb[sb2], random_state=0)[0]
        p10_b = precision_top10pct(yb[sb2], xb[sb2])
    else:
        auc_b = corr_b = mi_b = p10_b = np.nan

    rows_fi.append({
        "feature": col,
        "corr_labelA": corr_a,
        "mi_labelA": mi_a,
        "auc_labelA": auc_a,
        "prec10_labelA": p10_a,
        "corr_labelB": corr_b,
        "mi_labelB": mi_b,
        "auc_labelB": auc_b,
        "prec10_labelB": p10_b,
    })

fi_df = pd.DataFrame(rows_fi)
fi_df["score_A"] = fi_df["corr_labelA"].abs().fillna(0) + fi_df["mi_labelA"].fillna(0) + (fi_df["auc_labelA"].fillna(0) - 0.5).abs()
fi_df = fi_df.sort_values("score_A", ascending=False)

fi_df.head(20).to_parquet(OUTPUT_DIR / "feature_importance_top20.parquet", index=False)
display(fi_df.head(20))

fig, ax = plt.subplots(figsize=(10, 6))
subp = fi_df.head(20).sort_values("mi_labelA")
sns.barplot(data=subp, x="mi_labelA", y="feature", ax=ax, color="steelblue")
ax.set_title("Top 20 features by mutual information (label A)")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "bar_top20_mi_labelA.png", dpi=160)
plt.show()
"""
    )
)

cells.append(md_cell("## 9) Conditional probabilities"))

cells.append(
    code_cell(
        """LABEL_COND = "reached_plus_300_by_60s"
y = df[LABEL_COND].astype(int)

imb_cut = df["imbalance_top5"].quantile(0.75)
spread_med = df["spread_px"].median()
mom_pos = df["mid_return_5s"] > 0

conds = {
    "imbalance_top5 > q75": df["imbalance_top5"] > imb_cut,
    "high vol": df["vol_regime"] == "high",
    "high vol & imb>q75": (df["vol_regime"] == "high") & (df["imbalance_top5"] > imb_cut),
    "spread < median": df["spread_px"] < spread_med,
    "5s momentum > 0": mom_pos,
}

cond_rows = []
for name, m in conds.items():
    m = m.fillna(False)
    sub = y[m]
    cond_rows.append({"condition": name, "n": int(m.sum()), "p": float(sub.mean())})

cond_df = pd.DataFrame(cond_rows)
cond_df["baseline"] = float(y.mean())
from IPython.display import display

display(cond_df)
cond_df.to_parquet(OUTPUT_DIR / "conditional_probs.parquet", index=False)

fig, ax = plt.subplots(figsize=(9, 4))
sns.barplot(data=cond_df, x="p", y="condition", ax=ax, color="teal")
ax.axvline(float(y.mean()), color="crimson", ls="--", label="baseline")
ax.set_title("P(+300 ticks by 60s | condition)")
ax.legend()
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "bar_conditional_probs.png", dpi=160)
plt.show()
"""
    )
)

cells.append(md_cell("## 10) Автовыводы"))

cells.append(
    code_cell(
        """lines = []
lines.append(f"Instrument: {INSTRUMENT_LABEL}")
lines.append(f"Rows: {len(df):,} from {df['ts_event'].iloc[0]} to {df['ts_event'].iloc[-1]}")
lines.append(f"tick_size={tick_size}")

best = main_eda_df.sort_values(["threshold_ticks", "horizon_sec"], ascending=[True, True])
hi_share = best[(best["threshold_ticks"] >= 300) & (best["share_reached_plus"] > 0.05)]
lines.append("Horizons/thresholds with share_reached_plus>5% for ticks>=300:")
lines.append(hi_share[["horizon_sec", "threshold_ticks", "share_reached_plus"]].to_string(index=False))

lines.append("\\nVol regime counts:\\n" + df["vol_regime"].value_counts(dropna=False).to_string())

lines.append("\\nTop features (by |corr|+MI for label A):")
lines.append(fi_df.head(10)[["feature", "corr_labelA", "mi_labelA", "auc_labelA"]].to_string(index=False))

base_p = float(df["reached_plus_300_by_60s"].mean())
subp = float(dfb[LABEL_B].mean()) if len(dfb) else float("nan")
lines.append(f"\\nBaseline P(reached_plus_300_by_60s): {base_p:.4f}")
lines.append(f"Subsample P(TP first for TP300/SL150/60s): {subp:.4f}")

feat_pick = fi_df.head(8)["feature"].tolist()
lines.append("\\nSuggested first ML features: " + ", ".join(feat_pick))

summary_path = OUTPUT_DIR / "conclusions.txt"
summary_path.write_text("\\n".join(lines), encoding="utf-8")
print(summary_path.read_text(encoding="utf-8"))
"""
    )
)

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "cells": cells,
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print("Wrote", NB_PATH)
