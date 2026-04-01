# Building Baseline Modeling Dataset (RU)

Документация к ноутбуку `notebooks/models/001_build_modeling_dataset_baseline.ipynb`.

## Зачем нужен ноутбук

Чтобы перейти от EDA по reconstructed order book к **baseline modeling pipeline**:
- строим единый **dataset для short-horizon prediction**;
- для каждой строки задаём observation timestamp `ts_event` и берём **book features только из прошлого/на момент t**;
- добавляем **multi-horizon targets** на горизонтах:
  `100 / 150 / 200 / 250 / 500 / 1000 ms`;
- отдельно делаем **tick-based** и **profitability-aware** таргеты;
- добавляем вариант TP/SL labeling (порядок достижения TP/SL).

## Где лежит

- Ноутбук: `notebooks/models/001_build_modeling_dataset_baseline.ipynb`

## Входные данные

Нужны reconstructed parquet-файлы, которые уже лежат в проекте:

`data/reconstructed/<INST_ID>/grid100ms/<YYYY-MM-DD>/*.parquet`

Формат reconstructed parquet должен содержать минимум:
- `ts_event`
- `mid_px`
- `spread_px`
- `bid_px_01..bid_px_10`, `bid_sz_01..bid_sz_10`
- `ask_px_01..ask_px_10`, `ask_sz_01..ask_sz_10`
- служебные поля (обычно уже есть в parquet):
  - `inst_id`
  - `anchor_snapshot_ts`
  - `reconstruction_mode`

## Что является observation timestamp

В dataset каждая строка — это состояние стакана на момент времени **`ts_event = t`**.

- Features строятся из значений на `t` и на прошлых точках (`shift`/rolling по сетке).
- Targets строятся из **future state** (через `shift(-k)` на regular grid).

## Horizons: как выполняется alignment

Датасет рассчитан под **reconstructed grid 100ms** (`GRID_MS=100`).

Policy для горизонтов:
- для `h=150/250ms` таргеты строятся по first-observation policy на сетке:
  фактически это соответствует шагам `ceil(h / grid_ms)` (ceil-to-next-grid).

Это отражается в metadata `aligned_steps` / `aligned_horizon_ms`.

Если reconstructed данные не являются регулярной сеткой 100ms, pipeline может выдать ошибку (в коде есть sanity check регулярности).

## Leakage safeguards (обязательно)

Ноутбук и модули строятся по правилу:
- feature engineering не использует будущие значения;
- labels целиком зависят от future (shift(-k)).

Перед построением dataset выполняется sanity check и затем `dropna` по target-колонкам.
Это гарантирует отсутствие “случайной” утечки из-за отсутствующих future окон.

## Tick size: как используется

Tick size **не захардкожен** в логике.
По умолчанию pipeline:
- инферит tick size по наблюдаемым изменениям `mid_px`;
- сохраняет `tick_size` и notes в metadata.

При необходимости tick size можно задать явно в конфиге:
`TICK_SIZE = float(...)` вместо `None`.

## Cost model и profitability-aware targets

В profitability-aware target’ах используется maker/taker proxy:
- Long: entry на `best_bid_px_01(t)`, exit на `best_ask_px_01(t+H)`
- Short: entry на `best_ask_px_01(t)`, exit на `best_bid_px_01(t+H)`

Дальше считается gross/net потенциальная прибыль в **тиках**:
- net включает maker_fee_rate + taker_fee_rate,
- плюс `min_profit_ticks`.

Параметры задаются в начале ноутбука (`FEE_MODEL`, `GROSS_PROFIT_MIN_TICKS`).

## TP/SL style labeling

Допущения:
- entry — по топ-уровню:
  - long entry: `best_bid_px_01(t)`
  - short entry: `best_ask_px_01(t)`
- TP/SL оценивается по **mid_px excursion относительно entry**;
- в рамках holding window определяется порядок достижения:
  - `+1` → TP раньше SL
  - `-1` → SL раньше TP
  - `0` → timeout / ни одно не достигнуто (в текущей реализации: неявно через NaN/фильтрацию, а в конечном dataset сохраняются только строки с ненулевыми target).

Таргеты задаются наборами:
- `TP = (1,2,3)` ticks
- `SL = (1,2)` ticks
- `holding_ms = (100, 250, 500, 1000)`

## Какие families target строятся

Ноутбук создаёт несколько семейств target-колонок:
1) Raw directional target на mid:
   - `target_mid_dir_hXXX ∈ {-1,0,1}`
2) Tick threshold directional targets:
   - `target_mid_ge_1tick_hXXX`, `target_mid_le_minus_1tick_hXXX` и т.д.
3) Profitability-aware targets:
   - `target_gross_profitable_hXXX`
   - `target_net_profitable_hXXX`
4) TP/SL ordering labels:
   - `label_tp{TP}_sl{SL}_hold{H}_long`
   - `label_tp{TP}_sl{SL}_hold{H}_short`

Точный список сохраняется в metadata.

## Выходные артефакты

Dataset сохраняется в:

`data/processed/modeling/`

Файлы:
- Parquet:
  `<inst_id>_model_dataset_<GRID_MS>ms_baseline_<run_tag>.parquet`
- Metadata JSON:
  `<inst_id>_model_dataset_<GRID_MS>ms_baseline_<run_tag>_metadata.json`

Где `run_tag = <START_YYYYmmdd_HHMMSS>__<END_YYYYmmdd_HHMMSS>`.

metadata содержит:
- tick_size и notes
- aligned_steps/aligned_horizon_ms
- списки feature/target columns
- quality report (rows before/after filtering)
- target groups по семействам.

## Как запускать

1. Открой ноутбук:
   `notebooks/models/001_build_modeling_dataset_baseline.ipynb`
2. В начале ноутбука проверь конфиг:
   - `INST_ID`
   - `START`, `END`
   - `GRID_MS`, `DEPTH_K`
   - `HORIZONS_MS`, `LOOKBACK_MS`
   - fee model / TP-SL параметры
   - `TICK_SIZE` (если хочешь фиксировать вместо inference)
3. Запусти ячейки последовательно.
4. В конце проверь print-вывод путей до saved parquet + json.

## Производительность (важно)

Для периода 3 дня dataset может быть довольно большим.
Рекомендация:
- стартуй с меньшего окна, если pipeline запускается впервые;
- держи `GRID_MS=100` и глубину топа `DEPTH_K=10` как в baseline.

## Куда смотреть результат

- dataset parquet: `data/processed/modeling/`
- metadata json: рядом с parquet
- дальнейшая тренировка baseline модели должна использовать metadata для точного списка feature/target.

