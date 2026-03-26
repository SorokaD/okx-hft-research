# Восстановление стакана и корректная валидация (RU)

Практическая инструкция для себя "через месяц": как пересчитать стакан, как правильно валидировать качество восстановления и как не попасть в ловушку grid-based сравнения.

---

## 1) Кратко: что важно помнить

- Источник истины для валидации - **реальные snapshot timestamps** из `okx_raw.orderbook_snapshots`.
- Корректная проверка качества - **`snapshot_aligned`**:
  - состояние книги строится из snapshot + replay updates,
  - replay идет строго до `snapshot_ts`,
  - updates с `ts_event == snapshot_ts` включаются.
- Сравнение по `grid` (`grid_nearest`) полезно только как baseline/антипаттерн:
  - grid timestamp почти всегда не равен snapshot timestamp,
  - из-за этого появляются ложные mismatch.

---

## 2) Где что лежит

- Реконструкция стакана: `scripts/reconstruct_orderbook_to_parquet.py`
- Core-логика реконструкции: `research/reconstruction/reconstructor.py`
- Состояние книги в памяти: `research/reconstruction/book_state.py`
- Forensics/diagnostics старого типа: `scripts/diagnose_orderbook_forensics.py`
- Корректная валидация (snapshot-aligned + grid-nearest): `scripts/validate_orderbook_reconstruction.py`

Базовые output-папки:
- Реконструированные parquet: `data/reconstructed/...`
- Новая валидация: `data/diagnostics/orderbook_validation/...`

---

## 3) Подготовка окружения

Из корня репозитория:

```powershell
cd d:\tumar\okx-hft-research
.\.venv\Scripts\activate
```

Проверка, что скрипты импортируются:

```powershell
python -m py_compile scripts\reconstruct_orderbook_to_parquet.py
python -m py_compile scripts\validate_orderbook_reconstruction.py
```

---

## 4) Реконструкция стакана (raw -> parquet)

Ниже эталонный запуск для `BTC-USDT-SWAP`, полный день, `grid100ms`, сегментный режим `snapshot_to_snapshot`.

```powershell
python scripts\reconstruct_orderbook_to_parquet.py `
  --source raw `
  --schema okx_raw `
  --inst-id BTC-USDT-SWAP `
  --start "2026-03-22 00:00:00" `
  --end "2026-03-23 00:00:00" `
  --chunk-hours 1 `
  --mode grid `
  --grid-ms 100 `
  --depth 10 `
  --anchor-policy snapshot_to_snapshot `
  --snapshot-max-depth 0 `
  --output-dir data/reconstructed `
  --filename-prefix book
```

Ключевые флаги:
- `--anchor-policy snapshot_to_snapshot` - сброс состояния по snapshot->snapshot сегментам.
- `--snapshot-max-depth 0` - загружать полную глубину snapshot (без SQL hard-truncation по L10).
- `--mode grid --grid-ms 100` - выходной parquet на сетке (для downstream задач и baseline-сравнений).

---

## 5) Корректная валидация: snapshot-aligned

### 5.1 Что делает режим `snapshot_aligned`

Для каждого `snapshot_ts`:
- берет реальный snapshot как target,
- воспроизводит книгу replay-ем updates до `snapshot_ts` включительно,
- сравнивает reconstructed vs real на глубине `N`.

То есть сравнение выполняется **в одной и той же точке времени**.

### 5.2 Запуск

```powershell
python scripts\validate_orderbook_reconstruction.py `
  --mode snapshot_aligned `
  --inst-id BTC-USDT-SWAP `
  --schema okx_raw `
  --start "2026-03-22 00:00:00" `
  --end "2026-03-22 02:00:00" `
  --depth 10 `
  --worst-k 20 `
  --snapshot-limit 1000 `
  --output-dir data/diagnostics/orderbook_validation
```

`--snapshot-limit 0` = без лимита.

### 5.3 Артефакты

В папке вида:
`data/diagnostics/orderbook_validation/snapshot_aligned/<inst>_<start>__<end>/`

создаются:
- `validation_summary.json`
- `worst_points.csv`
- `forensic_cases/` (JSON по каждой worst-case точке)

---

## 6) Baseline для сравнения: grid_nearest

Нужен только чтобы показать, насколько хуже методология "ближайшая grid-точка".

Если окно >1 часа, удобно сначала склеить parquet:

```powershell
python -c "import pandas as pd; df=pd.concat([pd.read_parquet(r'd:\tumar\okx-hft-research\data\reconstructed\BTC-USDT-SWAP\grid100ms\2026-03-22\book_grid100ms_BTC-USDT-SWAP_2026-03-22_00-00-00__2026-03-22_01-00-00.parquet'), pd.read_parquet(r'd:\tumar\okx-hft-research\data\reconstructed\BTC-USDT-SWAP\grid100ms\2026-03-22\book_grid100ms_BTC-USDT-SWAP_2026-03-22_01-00-00__2026-03-22_02-00-00.parquet')], ignore_index=True).sort_values('ts_event'); df.to_parquet(r'd:\tumar\okx-hft-research\data\diagnostics\orderbook_validation\merged_tmp_00-02.parquet', index=False)"
```

Запуск baseline-валидации:

```powershell
python scripts\validate_orderbook_reconstruction.py `
  --mode grid_nearest `
  --inst-id BTC-USDT-SWAP `
  --schema okx_raw `
  --snapshot-table orderbook_snapshots `
  --reconstructed-parquet data/diagnostics/orderbook_validation/merged_tmp_00-02.parquet `
  --start "2026-03-22 00:00:00" `
  --end "2026-03-22 02:00:00" `
  --depth 10 `
  --worst-k 20 `
  --snapshot-limit 1000 `
  --output-dir data/diagnostics/orderbook_validation
```

---

## 7) Какие метрики смотреть в первую очередь

В `validation_summary.json`:

- `top1_bid_match_rate`
- `top1_ask_match_rate`
- `mean_px_diff`, `p95_px_diff`, `max_px_diff`
- `mean_sz_diff`, `p95_sz_diff`, `max_sz_diff`

Ожидаемая картина:
- `snapshot_aligned`: near-zero diff, top1 ~99-100%.
- `grid_nearest`: заметно хуже (ложные mismatch из-за временного сдвига).

---

## 8) Как интерпретировать worst cases

Смотри:
- `worst_points.csv` - быстрое ранжирование по тяжелым расхождениям.
- `forensic_cases/case_snapshot_*.json`:
  - `snapshot_ts`
  - reconstructed top levels
  - real snapshot top levels
  - `diff_breakdown` по каждому уровню и стороне.

Если в `snapshot_aligned` worst cases все равно большие:
- проверяй порядок updates внутри одинакового `ts_event`,
- проверяй корректность парсинга `bids_delta/asks_delta`,
- проверяй наличие пропусков snapshot/update в окне.

---

## 9) Частые ошибки и быстрые fixes

- Ошибка: сравнение с grid-рядом вместо snapshot-ts.  
  Fix: используй только `--mode snapshot_aligned` для quality verdict.

- Ошибка: "Object of type Timestamp is not JSON serializable".  
  Fix: перед `json.dumps` конвертировать timestamp в `isoformat()`.

- Ошибка: слишком мало validated points.  
  Fix: проверить окно времени (`--start/--end`) и `--snapshot-limit`.

- Ошибка: нет данных в parquet для multi-hour окна.  
  Fix: склеить часовые parquet в единый файл (как в разделе 6).

---

## 10) Мини-чеклист "вспомнить за 3 минуты"

1. Активировать `.venv`.
2. Пересчитать reconstruction (если нужно обновить данные).
3. Запустить `snapshot_aligned` валидацию.
4. Запустить `grid_nearest` baseline на том же окне.
5. Сравнить `validation_summary.json` по двум режимам.
6. Открыть `worst_points.csv` и 2-3 `forensic_cases`.
7. Делать вывод о качестве **только по snapshot_aligned**.

---

## 11) Быстрый "эталон" команд (копипаст)

```powershell
cd d:\tumar\okx-hft-research
.\.venv\Scripts\activate

# Snapshot-aligned (правильная валидация)
python scripts\validate_orderbook_reconstruction.py --mode snapshot_aligned --inst-id BTC-USDT-SWAP --schema okx_raw --start "2026-03-22 00:00:00" --end "2026-03-22 02:00:00" --depth 10 --worst-k 20 --snapshot-limit 1000 --output-dir data/diagnostics/orderbook_validation

# Grid-nearest (baseline/антипаттерн для контраста)
python scripts\validate_orderbook_reconstruction.py --mode grid_nearest --inst-id BTC-USDT-SWAP --schema okx_raw --snapshot-table orderbook_snapshots --reconstructed-parquet data/diagnostics/orderbook_validation/merged_tmp_00-02.parquet --start "2026-03-22 00:00:00" --end "2026-03-22 02:00:00" --depth 10 --worst-k 20 --snapshot-limit 1000 --output-dir data/diagnostics/orderbook_validation
```

---

Если через месяц нужно "просто понять, все ли ок", смотри сначала `validation_summary.json` из `snapshot_aligned` и только потом углубляйся в форензику.

