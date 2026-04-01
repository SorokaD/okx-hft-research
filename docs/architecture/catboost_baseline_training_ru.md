# CatBoost baseline: обучение на modeling dataset (short-horizon)

## Зачем этот ноутбук

Этот ноутбук (`notebooks/models/002_train_catboost_baseline.ipynb`) обучает **первую рабочую baseline-модель CatBoost** на уже собранном `model_dataset` и честно оценивает качество:

- ML-метрики на `valid` и `test`
- анализ качества предсказаний по confidence buckets
- калибровка вероятностей (reliability)
- простая trading-oriented proxy-оценка (для profitability-бинарных target’ов)
- сохранение артефактов обучения для дальнейшего сравнения

Ключевая цель: понять, есть ли вообще сигнал, и какие слабые места у baseline.

## Где лежат данные

- `data/processed/modeling/btc_usdt_swap_model_dataset_100ms_baseline_20260323_000000__20260325_235959_metadata.json`
- рядом с ним: `data/processed/modeling/btc_usdt_swap_model_dataset_100ms_baseline_20260323_000000__20260325_235959.parquet`

В ноутбуке paths можно менять в блоке `Config`.

## Что считается baseline-target’ом

По умолчанию при `HORIZON_MS = 100` включён priority:

1. `target_net_profitable_h100`
2. `target_gross_profitable_h100`
3. `target_mid_dir_h100`

Нужный target выбирается автоматически по наличию колонки в dataset.

Чтобы переобучить baseline на другом target/horizon, поменяйте в `Config`:

- `HORIZON_MS`
- `TARGET_PRIORITY`

Колонки, которые в dataset уже должны присутствовать (пример из метаданных):

- `target_net_profitable_h150`, `target_net_profitable_h200`
- `target_mid_ge_1tick_h100`, `target_mid_ge_2tick_h100`
- `label_tp2_sl1_hold500_long`

Если выбрали directional target (например `target_mid_dir_*`) — это будет **multiclass** (если в данных присутствуют 3 класса).

## Leakage safeguards (как защищаемся от будущей информации)

В feature set попадают только колонки из `metadata["feature_cols"]`, но они дополнительно фильтруются:

1. Исключаются `target_*`, `label_*`
2. Исключаются future-move признаки:
   - `mid_move_ticks_*`
   - `best_bid_move_ticks_*`
   - `best_ask_move_ticks_*`
3. Исключаются служебные/технические поля:
   - `ts_event`, `inst_id`, `anchor_snapshot_ts`, `reconstruction_mode`, `tick_size`

Split всегда **time-based**: без random shuffle.

## Time split

По умолчанию:

- train: первые `70%` строк (по возрастанию `ts_event`)
- valid: следующие `15%`
- test: последние `15%`

Ноутбук печатает:

- границы `start/end` для train/valid/test
- число строк в каждом split
- class distribution по выбранному target

## Метрики и отчёт

Для profitability-binary target’ов показываются:

- ROC-AUC, PR-AUC
- accuracy, balanced accuracy
- precision, recall, F1
- confusion matrix, classification report
- brier score (качество вероятностей)
- confidence buckets по `p(y=1)` (coverage + realized hit-rate)
- reliability (reliability-table + график)
- trading-oriented proxy (hit-rate vs coverage на тесте по порогам)

Для multiclass target’ов:

- accuracy, balanced accuracy
- macro/weighted F1
- confusion matrix и classification report
- confidence buckets по max-proba + top1 correctness rate

## Артефакты обучения (что сохраняется)

Все сохранения лежат в:

`data/artifacts/models/<run_id>/`

Состав:

- `model.cbm` — модель CatBoost
- `predictions_test.parquet` — `ts_event`, `y_true`, `y_pred`, `y_proba_pos`
- `confidence_buckets_test.csv`
- `calibration_bins_test.csv` — reliability table
- `calibration_curve_points_test.csv` — points reliability curve
- `predictions_top_confident_test.csv` — топ-N самых уверенных предсказаний на test
- `validation_metrics.json`, `test_metrics.json`
- `feature_importance_top30.csv`
- `model_metadata.json` — dataset/target/feature audit, split boundaries, параметры обучения и пути артефактов
- `data/artifacts/reports/<run_id>.md` — краткий human-readable отчёт

## Как запускать

1. Откройте `notebooks/models/002_train_catboost_baseline.ipynb`
2. В блоке `Config` убедитесь, что:
   - путь к dataset верный
   - `HORIZON_MS` и `TARGET_PRIORITY` соответствуют вашему плану
3. Запустите ноутбук сверху вниз.

## Важно

В текущем окружении `catboost` и `scikit-learn` могут быть не установлены. В таком случае код в ноутбуке упадёт на шаге обучения/вычисления метрик, и нужно будет установить пакеты в вашей Python-среде:

`pip install catboost scikit-learn`

## Рекомендации по использованию baseline

Если `test` метрики заметно проседают относительно `valid` — сигнал слабый или переобучение/нестабильность.

Если confidence buckets показывают, что high-confidence предсказания часто не подтверждаются — модель плохо калибрована или сигнал слабый.

В обоих случаях следующий шаг: попробовать другой target/horizon (или другую схему/feature set), но не “дотягивать” метрики тюнингом любой ценой.

