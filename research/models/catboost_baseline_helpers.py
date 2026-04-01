"""
Helpers for training and evaluating a CatBoost baseline on modeling datasets.

Note: CatBoost is imported lazily inside training functions to keep the repo
importable even when `catboost` isn't installed in the current environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.metrics import average_precision_score
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    _SKLEARN_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False


def _require_sklearn() -> None:
    if not _SKLEARN_AVAILABLE:
        raise ModuleNotFoundError(
            "scikit-learn is required for metric computation. Install it: `pip install scikit-learn`."
        )


@dataclass(frozen=True)
class SplitConfig:
    train_frac: float = 0.70
    valid_frac: float = 0.15
    test_frac: float = 0.15


@dataclass(frozen=True)
class TrainingConfig:
    iterations: int = 4000
    learning_rate: float = 0.03
    depth: int = 8
    l2_leaf_reg: float = 3.0
    random_seed: int = 42
    early_stopping_rounds: int = 200
    verbose: int = 200


def get_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def select_primary_target_column(
    df: pd.DataFrame,
    metadata: dict[str, Any],
    *,
    horizon_ms: int,
    priority: tuple[str, ...] = ("net", "gross", "mid_dir"),
) -> str:
    """
    priority elements:
    - 'net' -> target_net_profitable_h{h}
    - 'gross' -> target_gross_profitable_h{h}
    - 'mid_dir' -> target_mid_dir_h{h}
    """
    for p in priority:
        if p == "net":
            col = f"target_net_profitable_h{horizon_ms}"
        elif p == "gross":
            col = f"target_gross_profitable_h{horizon_ms}"
        elif p == "mid_dir":
            col = f"target_mid_dir_h{horizon_ms}"
        else:
            raise ValueError(f"Unknown priority element: {p}")
        if col in df.columns:
            return col
    raise ValueError(
        f"None of priority targets found for horizon {horizon_ms}. Missing columns. "
        f"Tried: {[('net' if p=='net' else p) for p in priority]}"
    )


def infer_target_type(series: pd.Series) -> Literal["binary", "multiclass"]:
    vals = series.dropna().unique()
    # multiclass for {-1,0,1} or more discrete values
    if len(vals) <= 2:
        return "binary"
    return "multiclass"


def normalize_multiclass_labels(y: pd.Series) -> tuple[np.ndarray, dict[int, int]]:
    """
    Map labels (e.g. {-1,0,1}) to {0..K-1} for CatBoost.
    """
    y_int = y.astype("int64").to_numpy()
    uniq = sorted(list(pd.Series(y_int).dropna().unique()))
    mapping = {int(v): i for i, v in enumerate(uniq)}
    y_mapped = np.vectorize(mapping.get)(y_int).astype("int32")
    return y_mapped, mapping


def build_feature_columns(
    df: pd.DataFrame,
    metadata: dict[str, Any],
    *,
    target_col: str,
) -> tuple[list[str], list[dict[str, str]]]:
    """
    Use metadata feature_cols as starting point, then exclude columns that should never
    be used as shortcuts/leakage.
    """
    meta_features = metadata.get("feature_cols")
    if not meta_features:
        meta_features = list(df.columns)

    target_prefixes = ("target_", "label_", "mid_move_ticks_", "best_bid_move_ticks_", "best_ask_move_ticks_")
    exclude_reasons: list[dict[str, str]] = []
    used: list[str] = []

    for c in meta_features:
        if c == target_col:
            exclude_reasons.append({"column": c, "reason": "target column"})
            continue

        if c in {"ts_event", "inst_id", "anchor_snapshot_ts", "reconstruction_mode", "tick_size"}:
            exclude_reasons.append({"column": c, "reason": "identifier / time column"})
            continue

        if any(c.startswith(pref) for pref in target_prefixes):
            exclude_reasons.append({"column": c, "reason": "future/label-derived column"})
            continue

        # Avoid any known "mode" leakage shortcuts (keep if desired later)
        used.append(c)

    excluded = [x["column"] for x in exclude_reasons]
    # include numeric filters: ensure all used columns are numeric
    used_numeric = [c for c in used if pd.api.types.is_numeric_dtype(df[c])]
    for c in used:
        if c not in used_numeric:
            exclude_reasons.append({"column": c, "reason": "non-numeric feature"})

    return used_numeric, exclude_reasons


def time_split(
    df: pd.DataFrame,
    *,
    ts_col: str,
    target_col: str,
    cfg: SplitConfig,
) -> dict[str, pd.DataFrame]:
    df_sorted = df.sort_values(ts_col).reset_index(drop=True)
    n = len(df_sorted)
    if n < 1000:
        raise ValueError("Dataset is too small for meaningful split")

    train_end = int(n * cfg.train_frac)
    valid_end = int(n * (cfg.train_frac + cfg.valid_frac))
    if train_end <= 0 or valid_end <= train_end or valid_end >= n:
        raise ValueError("Invalid split fractions")

    train = df_sorted.iloc[:train_end].copy()
    valid = df_sorted.iloc[train_end:valid_end].copy()
    test = df_sorted.iloc[valid_end:].copy()
    return {"train": train, "valid": valid, "test": test}


def time_split_boundaries(
    df: pd.DataFrame,
    *,
    ts_col: str,
    train_frac: float,
    valid_frac: float,
) -> dict[str, Any]:
    """
    Compute time boundaries for train/valid/test based on row order after sorting by ts_col.
    """
    df_sorted = df.sort_values(ts_col).reset_index(drop=True)
    n = len(df_sorted)
    if n == 0:
        raise ValueError(
            "Dataset is empty after preprocessing. "
            "Check dropna/subset rules and ensure target/ts_event are present."
        )
    train_end = int(n * train_frac)
    valid_end = int(n * (train_frac + valid_frac))
    if train_end <= 0 or valid_end <= train_end or valid_end >= n:
        raise ValueError(
            f"Invalid split fractions for n={n}: train_end={train_end}, valid_end={valid_end}. "
            "Adjust TRAIN_FRAC/VALID_FRAC."
        )
    return {
        "n": int(n),
        "train_end_idx": int(train_end),
        "valid_end_idx": int(valid_end),
        "train": {
            "start": pd.Timestamp(df_sorted[ts_col].iloc[0]),
            "end": pd.Timestamp(df_sorted[ts_col].iloc[train_end - 1]),
        },
        "valid": {
            "start": pd.Timestamp(df_sorted[ts_col].iloc[train_end]),
            "end": pd.Timestamp(df_sorted[ts_col].iloc[valid_end - 1]),
        },
        "test": {
            "start": pd.Timestamp(df_sorted[ts_col].iloc[valid_end]),
            "end": pd.Timestamp(df_sorted[ts_col].iloc[-1]),
        },
    }


def make_class_distribution_report(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    s = df[target_col].dropna()
    counts = s.value_counts().sort_index()
    shares = (counts / len(df)).to_dict()
    return {"counts": counts.to_dict(), "shares": {str(k): float(v) for k, v in shares.items()}, "n": int(len(df))}


def compute_binary_class_weights(y: np.ndarray) -> dict[int, float]:
    """
    Simple inverse-frequency class weights for CatBoost `class_weights`.
    """
    y = y.astype(int)
    vals, counts = np.unique(y, return_counts=True)
    if set(vals.tolist()) - {0, 1}:
        raise ValueError(f"Expected binary labels in {{0,1}}, got {vals.tolist()}")
    total = counts.sum()
    weights: dict[int, float] = {}
    for v, c in zip(vals, counts):
        # higher weight for minority class
        weights[int(v)] = float(total / (2 * c))
    # ensure both keys exist
    weights.setdefault(0, float(total / 2))
    weights.setdefault(1, float(total / 2))
    return weights


def train_catboost_binary(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    *,
    train_params: TrainingConfig,
    class_weights: dict[int, float] | None = None,
):
    try:
        from catboost import CatBoostClassifier  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "CatBoost is not installed. Install it in your environment: `pip install catboost`."
        ) from e

    model = CatBoostClassifier(
        iterations=train_params.iterations,
        learning_rate=train_params.learning_rate,
        depth=train_params.depth,
        l2_leaf_reg=train_params.l2_leaf_reg,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=train_params.random_seed,
        verbose=train_params.verbose,
        od_type="Iter",
        od_wait=train_params.early_stopping_rounds,
        use_best_model=True,
        class_weights=class_weights,
    )
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
    return model


def train_catboost_multiclass(
    X_train: pd.DataFrame,
    y_train_mapped: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid_mapped: np.ndarray,
    *,
    train_params: TrainingConfig,
    num_classes: int,
):
    try:
        from catboost import CatBoostClassifier  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "CatBoost is not installed. Install it in your environment: `pip install catboost`."
        ) from e

    model = CatBoostClassifier(
        iterations=train_params.iterations,
        learning_rate=train_params.learning_rate,
        depth=train_params.depth,
        l2_leaf_reg=train_params.l2_leaf_reg,
        loss_function="MultiClass",
        eval_metric=f"MultiClass",  # CatBoost provides its own, this is a safe choice
        random_seed=train_params.random_seed,
        verbose=train_params.verbose,
        od_type="Iter",
        od_wait=train_params.early_stopping_rounds,
        use_best_model=True,
        # class_weights could be added if needed
    )
    model.fit(X_train, y_train_mapped, eval_set=(X_valid, y_valid_mapped))
    return model


def make_confidence_bucket_report(
    y_true: np.ndarray,
    y_proba_pos: np.ndarray,
    *,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    bins = [0.0] + thresholds + [1.0]
    bucket_names = []
    rows = []
    for i in range(len(bins) - 1):
        lo = bins[i]
        hi = bins[i + 1]
        name = f"{lo:.1f}–{hi:.1f}"
        bucket_names.append(name)
        mask = (y_proba_pos >= lo) & (y_proba_pos < hi) if i < len(bins) - 2 else (y_proba_pos >= lo) & (y_proba_pos <= hi)
        if not np.any(mask):
            continue
        yb = y_true[mask]
        precision = float(yb.mean())
        rows.append(
            {
                "bucket": name,
                "n": int(mask.sum()),
                "coverage": float(mask.mean()),
                "hit_rate_realized": precision,
            }
        )
    return pd.DataFrame(rows)


def reliability_table(
    y_true: np.ndarray,
    y_proba_pos: np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Reliability-style table by probability bins.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    for i in range(n_bins):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i < n_bins - 1:
            mask = (y_proba_pos >= lo) & (y_proba_pos < hi)
        else:
            mask = (y_proba_pos >= lo) & (y_proba_pos <= hi)
        if not np.any(mask):
            continue
        rows.append(
            {
                "p_bin_lo": lo,
                "p_bin_hi": hi,
                "n": int(mask.sum()),
                "mean_pred_proba": float(y_proba_pos[mask].mean()),
                "realized_positive_rate": float(y_true[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def get_feature_importance(
    model: Any,
    feature_cols: list[str],
    *,
    top_n: int = 30,
) -> pd.DataFrame:
    """
    CatBoost built-in feature importance (gain-based) top-N.
    """
    importances = model.get_feature_importance(type="FeatureImportance")
    fi = pd.DataFrame({"feature": feature_cols, "importance": importances})
    fi = fi.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
    return fi


def save_predictions_parquet(
    pred_path: Path,
    *,
    ts_event: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba_pos: np.ndarray,
) -> None:
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(
        {
            "ts_event": pd.to_datetime(ts_event, utc=True, errors="coerce"),
            "y_true": y_true.astype(int),
            "y_pred": y_pred.astype(int),
            "y_proba_pos": y_proba_pos.astype(float),
        }
    )
    out.to_parquet(pred_path, index=False)


def evaluate_binary_classifier(
    model: Any,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
):
    _require_sklearn()

    def _eval(X: pd.DataFrame, y: np.ndarray) -> dict[str, Any]:
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)

        roc_auc = roc_auc_score(y, proba) if len(np.unique(y)) == 2 else np.nan
        pr_auc = average_precision_score(y, proba) if len(np.unique(y)) == 2 else np.nan
        acc = accuracy_score(y, pred)
        bal_acc = balanced_accuracy_score(y, pred)
        prec = precision_score(y, pred, zero_division=0)
        rec = recall_score(y, pred, zero_division=0)
        f1 = f1_score(y, pred, zero_division=0)
        cm = confusion_matrix(y, pred)
        cr = classification_report(y, pred, output_dict=True, zero_division=0)
        brier = brier_score_loss(y, proba)

        return {
            "roc_auc": float(roc_auc) if roc_auc == roc_auc else None,
            "pr_auc": float(pr_auc) if pr_auc == pr_auc else None,
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "confusion_matrix": cm.tolist(),
            "classification_report": cr,
            "brier_score": float(brier),
        }

    valid_metrics = _eval(X_valid, y_valid)
    test_metrics = _eval(X_test, y_test)

    return valid_metrics, test_metrics


def evaluate_multiclass_classifier(
    model: Any,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    *,
    class_names: list[str] | None = None,
    confidence_thresholds: list[float] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    """
    Multiclass metrics + confidence bucket report based on `max proba`.
    """
    _require_sklearn()

    if confidence_thresholds is None:
        confidence_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    def _eval(X: pd.DataFrame, y: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame]:
        proba = model.predict_proba(X)
        pred = proba.argmax(axis=1)

        acc = accuracy_score(y, pred)
        bal_acc = balanced_accuracy_score(y, pred)
        macro_f1 = f1_score(y, pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y, pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y, pred)
        cr = classification_report(
            y,
            pred,
            output_dict=True,
            zero_division=0,
            target_names=class_names if class_names is not None else None,
        )

        max_proba = proba.max(axis=1)
        bucket_df = make_confidence_bucket_report_multiclass(
            y_true=y,
            y_pred=pred,
            y_proba_max=max_proba,
            confidence_thresholds=confidence_thresholds,
        )

        metrics = {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "confusion_matrix": cm.tolist(),
            "classification_report": cr,
        }
        return metrics, bucket_df

    valid_metrics, valid_bucket_df = _eval(X_valid, y_valid)
    test_metrics, test_bucket_df = _eval(X_test, y_test)
    # Return test bucket report primarily.
    return valid_metrics, test_metrics, test_bucket_df


def make_confidence_bucket_report_multiclass(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba_max: np.ndarray,
    *,
    confidence_thresholds: list[float],
) -> pd.DataFrame:
    """
    For multiclass: bucket by model confidence (= max class probability),
    and report top1 correctness rate per bucket.
    """
    bins = [0.0] + list(confidence_thresholds) + [1.0]
    rows: list[dict[str, Any]] = []
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    for i in range(len(bins) - 1):
        lo = float(bins[i])
        hi = float(bins[i + 1])
        if i < len(bins) - 2:
            mask = (y_proba_max >= lo) & (y_proba_max < hi)
        else:
            mask = (y_proba_max >= lo) & (y_proba_max <= hi)
        if not np.any(mask):
            continue
        rows.append(
            {
                "bucket": f"{lo:.1f}–{hi:.1f}",
                "n": int(mask.sum()),
                "coverage": float(mask.mean()),
                "top1_correct_rate": float((y_pred[mask] == y_true[mask]).mean()),
            }
        )
    return pd.DataFrame(rows)


def calibration_curve_data(
    y_true: np.ndarray,
    y_proba_pos: np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Simple reliability curve points for binary targets.
    """
    _require_sklearn()
    frac_pos, mean_pred = calibration_curve(y_true, y_proba_pos, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame(
        {
            "p_bin_mean_pred": mean_pred.astype(float),
            "observed_frac_pos": frac_pos.astype(float),
        }
    )


def trading_evaluation_profitability(
    y_true: np.ndarray,
    y_proba_pos: np.ndarray,
    *,
    thresholds: list[float],
) -> pd.DataFrame:
    rows = []
    total = len(y_true)
    for thr in thresholds:
        mask = y_proba_pos >= thr
        if not np.any(mask):
            rows.append(
                {
                    "prob_threshold": thr,
                    "coverage": 0.0,
                    "hit_rate_realized": np.nan,
                }
            )
            continue
        coverage = float(mask.mean())
        hit_rate = float(y_true[mask].mean())
        rows.append(
            {
                "prob_threshold": thr,
                "coverage": coverage,
                "hit_rate_realized": hit_rate,
                "expected_value_proxy": hit_rate,  # proxy since target is already profitability indicator
                "n_signals": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("prob_threshold")

