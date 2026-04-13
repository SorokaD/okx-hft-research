import json
import pathlib
import re

p = pathlib.Path(r'D:/tumar/okx-hft-research/notebooks/research/005_eda_targets_predictability.ipynb')
nb = json.loads(p.read_text(encoding='utf-8'))

# ---- cell 2: replace run_baseline_model ----
cell2 = nb['cells'][2]
s2 = ''.join(cell2['source'])
new_run = '''def run_baseline_model(
    df: pd.DataFrame,
    *,
    target_col: str,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    feature_columns = exclude_raw_price_features(list(feature_columns))
    if not feature_columns:
        return pd.DataFrame(), {}

    y = pd.to_numeric(df[target_col], errors="coerce")
    X = df[feature_columns].copy()
    mask = y.isin([0, 1])
    X = X.loc[mask]
    y = y.loc[mask].astype(int)
    if len(X) < 1000:
        return pd.DataFrame(), {}

    # keep the same time-based split (no shuffle)
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        return pd.DataFrame(), {}

    rows = []

    # 1) Logistic regression baseline (same split/features/prep)
    logreg = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=300, random_state=RANDOM_SEED)),
        ]
    )
    logreg.fit(X_train, y_train)
    proba = logreg.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    rows.append(
        {
            "target": target_col,
            "model": "logreg",
            "roc_auc": roc_auc_score(y_test, proba),
            "pr_auc": average_precision_score(y_test, proba),
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "positive_rate": float(y_test.mean()),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        }
    )

    # 2) CatBoost for every target: class_weights + early stopping
    from catboost import CatBoostClassifier

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    if n_pos <= 0 or n_neg <= 0:
        return pd.DataFrame(rows), {}
    weight_0 = 1.0
    weight_1 = float(n_neg / n_pos)

    cat_params = {
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "iterations": 1200,
        "learning_rate": 0.03,
        "depth": 5,
        "l2_leaf_reg": 10,
        "subsample": 0.8,
        "random_strength": 1.0,
        "bootstrap_type": "Bernoulli",
        "min_data_in_leaf": 100,
        "leaf_estimation_iterations": 5,
        "grow_policy": "SymmetricTree",
        "verbose": 100,
        "random_seed": 42,
        "allow_writing_files": False,
        "thread_count": -1,
        "class_weights": [weight_0, weight_1],
    }

    # keep the same preparation logic: median imputation
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    cat = CatBoostClassifier(**cat_params)
    cat.fit(
        X_train_imp,
        y_train,
        eval_set=(X_test_imp, y_test),
        use_best_model=True,
        early_stopping_rounds=100,
    )

    proba = cat.predict_proba(X_test_imp)[:, 1]
    pred = (proba >= 0.5).astype(int)
    try:
        best_iter = int(cat.get_best_iteration())
    except Exception:
        best_iter = -1

    rows.append(
        {
            "target": target_col,
            "model": "catboost",
            "roc_auc": roc_auc_score(y_test, proba),
            "pr_auc": average_precision_score(y_test, proba),
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "positive_rate": float(y_test.mean()),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "weight_1": weight_1,
            "best_iteration": best_iter,
        }
    )

    model_info = {
        "model_name": "catboost",
        "model": cat,
        "imputer": imputer,
        "feature_columns": feature_columns,
        "X_train": X_train,
        "y_train": y_train,
    }

    return pd.DataFrame(rows), model_info
'''
s2 = re.sub(
    r'def run_baseline_model\([\s\S]*?return pd\.DataFrame\(rows\), feature_importance_model or \{\}\n',
    new_run,
    s2,
    count=1,
)
cell2['source'] = s2.splitlines(keepends=True)

# ---- cell 4: replace loader ----
cell4 = nb['cells'][4]
s4 = ''.join(cell4['source'])
new_loader = '''def load_wide_merged_parquet(path: Path, *, max_rows: int | None = None) -> pd.DataFrame:
    """Load wide merged parquet using Arrow concat to avoid pandas block consolidation OOM."""

    pf = pq.ParquetFile(path)
    tables: list[pa.Table] = []
    total = 0
    for rg in range(pf.num_row_groups):
        if max_rows is not None:
            need = max_rows - total
            if need <= 0:
                break

        sub = pf.read_row_group(rg)
        sub = _table_downcast_float64_to_float32(sub)

        if max_rows is not None:
            need = max_rows - total
            if sub.num_rows > need:
                sub = sub.slice(0, need)

        total += sub.num_rows
        tables.append(sub)

        if max_rows is not None and total >= max_rows:
            break

    if not tables:
        return pd.DataFrame()

    try:
        big = pa.concat_tables(tables, promote_options="default")
    except TypeError:
        big = pa.concat_tables(tables)

    del tables
    gc.collect()

    to_pandas_kwargs: dict[str, object] = {"types_mapper": pd.ArrowDtype}
    try:
        df_local = big.to_pandas(self_destruct=True, **to_pandas_kwargs)
    except TypeError:
        df_local = big.to_pandas(**to_pandas_kwargs)

    del big
    gc.collect()
    return df_local
'''
s4 = re.sub(
    r'def load_wide_merged_parquet\(path: Path, \*, max_rows: int \| None = None\) -> pd\.DataFrame:\n[\s\S]*?\n\n\ndf = load_wide_merged_parquet\(DATA_PATH, max_rows=PARQUET_LOAD_MAX_ROWS\)\n',
    new_loader + '\n\ndf = load_wide_merged_parquet(DATA_PATH, max_rows=PARQUET_LOAD_MAX_ROWS)\n',
    s4,
    count=1,
)
cell4['source'] = s4.splitlines(keepends=True)

# ---- feature importance cell compatibility (model instead of pipeline) ----
for c in nb['cells']:
    src = ''.join(c.get('source', []))
    if 'pipe = model_info["pipeline"]' in src:
        src = src.replace('    pipe = model_info["pipeline"]\n', '')
        src = src.replace('    fitted_model = pipe.named_steps["model"]\n', '    fitted_model = model_info["model"]\n')
        c['source'] = src.splitlines(keepends=True)

p.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding='utf-8')
print('patched_all')
