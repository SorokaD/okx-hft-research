"""Compare CatBoost baselines under different feature groups for one binary target.

Example:
  python -m research.experiments.compare_feature_groups \\
    --parquet-path outputs/features/btc_usdt_swap_features_w3600.parquet \\
    --target short_hit_down_150t_before_up_within_30s \\
    --output-dir outputs/experiments/feature_group_compare_short150

Feature groups (see docstring of ``feature_group_columns``):
  A SHORT_ONLY, B SHORT_PLUS_300, C SHORT_PLUS_3600, D ALL_WITHOUT_PRICE, E ALL_FEATURES
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

LOGGER = logging.getLogger(__name__)

META_COLUMNS = frozenset(
    {
        "ts_event",
        "inst_id",
        "anchor_snapshot_ts",
        "reconstruction_mode",
    }
)

# Binary / regression targets in the modeling parquet (exclude from X).
_TARGET_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(long|short)_hit_.*"),
    re.compile(r"^(long|short)_max_(favorable|adverse)_excursion_ticks_within_\d+s$"),
)

# Horizons > 30s wall-clock in feature names (exclude from SHORT_ONLY).
_LONG_HORIZON_SUFFIX = re.compile(r"_(?:60|300|900|1800|3600)s$")

# Regime / persistence features built from longer context; keep out of SHORT_ONLY.
_REGIME_LIKE = frozenset(
    {
        "spread_regime",
        "volatility_regime",
        "update_regime",
        "seconds_since_large_move",
    }
)

_DOMINANCE_KEYS = ("mid_px", "realized_vol_3600s", "mid_return_3600s", "realized_vol_300s", "mid_return_300s")


def _is_target_column(name: str) -> bool:
    return any(p.match(name) for p in _TARGET_RES)


def infer_numeric_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    """All numeric columns except meta, target, and any other target-like columns."""

    out: list[str] = []
    for c in df.columns:
        if c in META_COLUMNS or c == target_col:
            continue
        if _is_target_column(c):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        out.append(c)
    return sorted(out)


def feature_group_columns(
    all_features: list[str],
    *,
    group: str,
) -> list[str]:
    """Return feature names for experiment group A–E (intersected with dataset later)."""

    af = set(all_features)

    def only_300s(c: str) -> bool:
        return c.endswith("_300s")

    def only_3600s(c: str) -> bool:
        return c.endswith("_3600s")

    # A: no long horizons, no mid_px, no regime-like aggregates.
    short_only = [
        c
        for c in all_features
        if c != "mid_px"
        and not _LONG_HORIZON_SUFFIX.search(c)
        and c not in _REGIME_LIKE
    ]

    if group == "SHORT_ONLY":
        return short_only
    if group == "SHORT_PLUS_300":
        return sorted(set(short_only) | {c for c in all_features if only_300s(c)})
    if group == "SHORT_PLUS_3600":
        return sorted(set(short_only) | {c for c in all_features if only_3600s(c)})
    if group == "ALL_WITHOUT_PRICE":
        return [c for c in all_features if c != "mid_px"]
    if group == "ALL_FEATURES":
        return list(all_features)
    raise ValueError(f"Unknown group: {group!r}")


def intersect_features(requested: Iterable[str], available: list[str]) -> list[str]:
    av = set(available)
    return [c for c in requested if c in av]


@dataclass
class ModelResult:
    model_name: str
    n_features: int
    features_used: list[str]
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    threshold: float
    positive_pred_rate: float
    importance_top20: list[tuple[str, float]] = field(default_factory=list)
    importance_top10: list[tuple[str, float]] = field(default_factory=list)


def _best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float, float, float, float]:
    """Grid search threshold on validation for max F1; return t, p, r, f1, pos_rate."""

    best_t, best_f1, best_p, best_r = 0.5, 0.0, 0.0, 0.0
    for t in np.linspace(0.001, 0.999, 399):
        pred = (proba >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            best_p = precision_score(y_true, pred, zero_division=0)
            best_r = recall_score(y_true, pred, zero_division=0)
    pos_rate = float((proba >= best_t).mean())
    return best_t, best_p, best_r, best_f1, pos_rate


def train_catboost_group(
    df: pd.DataFrame,
    *,
    target_col: str,
    feature_cols: list[str],
    model_name: str,
    random_seed: int,
) -> ModelResult:
    if len(feature_cols) < 2:
        raise ValueError(f"{model_name}: need >= 2 features, got {len(feature_cols)}")

    y = pd.to_numeric(df[target_col], errors="coerce")
    X = df[feature_cols]
    mask = y.isin([0, 1])
    X = X.loc[mask]
    y = y.loc[mask].astype(int)
    if len(X) < 2000:
        raise ValueError(f"{model_name}: too few rows after filter: {len(X)}")

    split_idx = int(len(X) * 0.7)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    if y_train.nunique() < 2 or y_val.nunique() < 2:
        raise ValueError(f"{model_name}: need both classes in train and val")

    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError("Install catboost: pip install catboost") from exc

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                CatBoostClassifier(
                    depth=6,
                    iterations=400,
                    learning_rate=0.05,
                    random_seed=random_seed,
                    verbose=False,
                    allow_writing_files=False,
                    thread_count=-1,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_val)[:, 1]

    roc = float(roc_auc_score(y_val, proba))
    pr = float(average_precision_score(y_val, proba))
    thr, prec, rec, f1, pos_rate = _best_f1_threshold(y_val.to_numpy(), proba)

    fitted = pipe.named_steps["model"]
    # After imputation, feature order matches feature_cols
    names = list(feature_cols)
    if hasattr(fitted, "feature_importances_"):
        imp = np.asarray(fitted.feature_importances_, dtype=float)
    else:
        imp = np.zeros(len(names))
    order = np.argsort(-imp)
    top_idx = order[:20]
    top20 = [(names[i], float(imp[i])) for i in top_idx]
    top10 = top20[:10]

    return ModelResult(
        model_name=model_name,
        n_features=len(feature_cols),
        features_used=feature_cols,
        roc_auc=roc,
        pr_auc=pr,
        precision=prec,
        recall=rec,
        f1=f1,
        threshold=float(thr),
        positive_pred_rate=pos_rate,
        importance_top20=top20,
        importance_top10=top10,
    )


def _plt():
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ImportError:
        return None
    return plt


def plot_importance_bar(result: ModelResult, out_path: Path) -> bool:
    plt = _plt()
    if plt is None:
        return False
    names, vals = zip(*result.importance_top20) if result.importance_top20 else ([], [])
    if not names:
        return False
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, vals, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"Top-20 features — {result.model_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def plot_pr_auc_compare(results: list[ModelResult], out_path: Path) -> bool:
    plt = _plt()
    if plt is None:
        return False
    fig, ax = plt.subplots(figsize=(9, 4))
    names = [r.model_name for r in results]
    prs = [r.pr_auc for r in results]
    x = np.arange(len(names))
    ax.bar(x, prs, color="steelblue")
    ax.set_ylabel("PR AUC (validation)")
    ax.set_title("PR AUC by feature group")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=22, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def dominance_note(top10: list[tuple[str, float]]) -> str:
    top3 = {n for n, _ in top10[:3]}
    hits = [k for k in _DOMINANCE_KEYS if k in top3]
    if hits:
        return f"Top-3 contains long/price drivers: {hits}"
    return "Top-3 has no mid_px / 300s / 3600s dominance pattern"


def build_text_report(
    results: dict[str, ModelResult],
    *,
    target_col: str,
    row_a: ModelResult,
    row_b: ModelResult,
    row_c: ModelResult,
    row_d: ModelResult,
    row_e: ModelResult,
) -> str:
    lines: list[str] = []
    lines.append(f"Target: `{target_col}`")
    lines.append("")
    lines.append("## Metrics (validation, chronological 70/30)")
    lines.append("")
    lines.append("| model_name | n_features | roc_auc | pr_auc | precision | recall | f1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for key in ("SHORT_ONLY", "SHORT_PLUS_300", "SHORT_PLUS_3600", "ALL_WITHOUT_PRICE", "ALL_FEATURES"):
        r = results[key]
        lines.append(
            f"| {r.model_name} | {r.n_features} | {r.roc_auc:.4f} | {r.pr_auc:.4f} | "
            f"{r.precision:.4f} | {r.recall:.4f} | {r.f1:.4f} |"
        )
    lines.append("")

    d_pr, e_pr = row_d.pr_auc, row_e.pr_auc
    d_f1, e_f1 = row_d.f1, row_e.f1
    pr_delta = e_pr - d_pr
    f1_delta = e_f1 - d_f1
    lines.append("## Effect of `mid_px` (E vs D)")
    lines.append("")
    if pr_delta > 0.002:
        lines.append(
            f"Removing absolute price **hurts** validation PR AUC by {pr_delta:.4f} "
            f"(F1 delta {f1_delta:+.4f}): the model uses level information."
        )
    elif pr_delta < -0.002:
        lines.append(
            f"Removing `mid_px` **improves** PR AUC by {-pr_delta:.4f} (F1 delta {f1_delta:+.4f}): "
            f"price level may be harmful or redundant."
        )
    else:
        lines.append(
            f"`mid_px` has **little effect** on PR AUC (delta {pr_delta:+.4f}, F1 delta {f1_delta:+.4f})."
        )
    lines.append("")

    lines.append("## Long horizons (B vs A, C vs A)")
    lines.append("")
    lines.append(f"- PR AUC gain from **+300s** features: {row_b.pr_auc - row_a.pr_auc:+.4f} (F1 {row_b.f1 - row_a.f1:+.4f})")
    lines.append(f"- PR AUC gain from **+3600s** features: {row_c.pr_auc - row_a.pr_auc:+.4f} (F1 {row_c.f1 - row_a.f1:+.4f})")
    lines.append("")

    dom_keys = ("mid_px", "realized_vol_3600s", "mid_return_3600s", "realized_vol_300s", "mid_return_300s")
    top3_set = {row_e.importance_top10[i][0] for i in range(min(3, len(row_e.importance_top10)))}
    if dom_keys & top3_set and row_e.pr_auc > row_a.pr_auc + 0.01:
        lines.append(
            "**Observation:** ALL_FEATURES beats SHORT_ONLY materially while top-3 importance "
            f"includes {sorted(dom_keys & top3_set)} — gains may come from regime/level, not local book."
        )
        lines.append("")
        lines.append(
            "**Recommendation:** Consider a two-stage setup: (1) regime filter using 300s/3600s "
            "(and possibly coarse price context), (2) separate entry model on 1–30s microstructure only."
        )
    else:
        lines.append(
            "**Observation:** Best full model does not obviously hinge only on long horizons/price in top-3; "
            "review barplots to confirm."
        )
    lines.append("")

    # Pick "most robust": prefer high PR AUC among A,B,C with low reliance on long features in top-10
    def score_robust(r: ModelResult) -> tuple[float, int]:
        pen = 0
        for n, _ in r.importance_top10:
            if _LONG_HORIZON_SUFFIX.search(n) or n == "mid_px":
                pen += 1
        return (r.pr_auc, -pen)

    candidates = [row_a, row_b, row_c]
    best_rob = max(candidates, key=score_robust)
    lines.append("## Suggested production stance (heuristic)")
    lines.append("")
    lines.append(
        f"- **Relatively robust group:** `{best_rob.model_name}` (PR AUC {best_rob.pr_auc:.4f}, "
        f"top-10 long/price count penalty considered)."
    )
    lines.append(
        f"- **Features to keep for local model:** prioritize SHORT_ONLY set; add 300s/3600s only if "
        f"offline monitoring shows stable lift on rolling forward tests."
    )
    lines.append(
        "- **Likely overfitting drivers:** `mid_px`, `realized_vol_3600s`, `mid_return_3600s` when they "
        "dominate importance but ablations show most lift from horizons > 30s."
    )
    lines.append("")

    for r in results.values():
        lines.append(f"### {r.model_name} — top-10 importance")
        for name, val in r.importance_top10:
            lines.append(f"- {name}: {val:.6f}")
        lines.append(f"  - _{dominance_note(r.importance_top10)}_")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet-path",
        type=Path,
        default=None,
        help="Features parquet (default: repo outputs/features/btc_usdt_swap_features_w3600.parquet)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="short_hit_down_150t_before_up_within_30s",
        help="Binary target column name",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write CSV, plots, report (default: outputs/experiments/feature_group_<target>)",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="Do not drop rows with NaN mid_return_3600s (default: trim like notebook 005).",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(message)s")

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    parquet_path = args.parquet_path or (root / "outputs" / "features" / "btc_usdt_swap_features_w3600.parquet")
    if not parquet_path.is_file():
        raise SystemExit(f"Parquet not found: {parquet_path}")

    safe_target = re.sub(r"[^\w\-.]+", "_", args.target)[:80]
    out_dir = args.output_dir or (root / "outputs" / "experiments" / f"feature_group_compare_{safe_target}")
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading %s", parquet_path)
    df = pd.read_parquet(parquet_path)
    if "ts_event" in df.columns:
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
        df = df.sort_values("ts_event").reset_index(drop=True)

    trim_col = "mid_return_3600s"
    if not args.no_trim and trim_col in df.columns:
        n0 = len(df)
        df = df[df[trim_col].notna()].reset_index(drop=True)
        LOGGER.info("Trimmed %d rows without %s", n0 - len(df), trim_col)
    elif args.no_trim:
        LOGGER.info("Skipping lookback trim (--no-trim)")

    target_col = args.target
    if target_col not in df.columns:
        raise SystemExit(f"Target column missing: {target_col}")

    all_features = infer_numeric_feature_columns(df, target_col)
    LOGGER.info("Numeric candidate features: %d", len(all_features))

    groups = {
        "SHORT_ONLY": "A_SHORT_ONLY",
        "SHORT_PLUS_300": "B_SHORT_PLUS_300",
        "SHORT_PLUS_3600": "C_SHORT_PLUS_3600",
        "ALL_WITHOUT_PRICE": "D_ALL_WITHOUT_PRICE",
        "ALL_FEATURES": "E_ALL_FEATURES",
    }

    results: dict[str, ModelResult] = {}
    for key, label in groups.items():
        requested = feature_group_columns(all_features, group=key)
        use_cols = intersect_features(requested, all_features)
        LOGGER.info("%s: requested %d -> available %d", label, len(requested), len(use_cols))
        if len(use_cols) < 2:
            LOGGER.warning("Skipping %s: too few columns", label)
            continue
        results[key] = train_catboost_group(
            df,
            target_col=target_col,
            feature_cols=use_cols,
            model_name=label,
            random_seed=args.random_seed,
        )

    if len(results) < 5:
        raise SystemExit("Not all 5 models trained; check feature availability and logs.")

    rows = []
    for key in ("SHORT_ONLY", "SHORT_PLUS_300", "SHORT_PLUS_3600", "ALL_WITHOUT_PRICE", "ALL_FEATURES"):
        r = results[key]
        rows.append(
            {
                "model_key": key,
                "model_name": r.model_name,
                "n_features": r.n_features,
                "roc_auc": r.roc_auc,
                "pr_auc": r.pr_auc,
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1,
                "threshold": r.threshold,
                "positive_pred_rate": r.positive_pred_rate,
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_path = out_dir / "metrics_comparison.csv"
    summary_df.to_csv(summary_path, index=False)
    LOGGER.info("Wrote %s", summary_path)

    plotted = 0
    for r in results.values():
        if plot_importance_bar(r, out_dir / f"importance_top20_{r.model_name}.png"):
            plotted += 1
    if plotted == 0:
        LOGGER.warning("matplotlib not installed: skipped bar plots (pip install matplotlib)")

    if not plot_pr_auc_compare(list(results.values()), out_dir / "pr_auc_by_group.png"):
        LOGGER.warning("matplotlib not installed: skipped PR AUC comparison plot")

    payload = {
        target_col: {
            r.model_name: {
                "top20": r.importance_top20,
                "metrics": {
                    "roc_auc": r.roc_auc,
                    "pr_auc": r.pr_auc,
                    "f1": r.f1,
                },
            }
            for r in results.values()
        }
    }
    (out_dir / "importance_json.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = build_text_report(
        results,
        target_col=target_col,
        row_a=results["SHORT_ONLY"],
        row_b=results["SHORT_PLUS_300"],
        row_c=results["SHORT_PLUS_3600"],
        row_d=results["ALL_WITHOUT_PRICE"],
        row_e=results["ALL_FEATURES"],
    )
    report_path = out_dir / "REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    LOGGER.info("Wrote %s", report_path)

    print("\n" + "=" * 72)
    print(report)
    print("=" * 72)
    print(f"\nArtifacts under: {out_dir}")


if __name__ == "__main__":
    main()
