# Источники данных

## TimescaleDB (okx_hft)

Основной источник — PostgreSQL/TimescaleDB со схемами `okx_raw` и `okx_core`.

### Таблицы

| Схема | Таблица | Описание |
|-------|---------|----------|
| okx_raw | orderbook_snapshots | Снимки стакана (wide: bid/ask по уровням) |
| okx_raw | orderbook_updates | Инкрементальные обновления стакана |
| okx_raw | trades | Сделки |
| okx_core | fact_orderbook_l10_snapshot | Агрегированный L10 стакан (bid_px_01..bid_sz_10, ask_px_01..ask_sz_10, mid_px, spread_px) |
| okx_core | fact_trades_tick | Тики сделок |
| okx_core | fact_mark_price_tick | Mark price |

### Подключение

Переменные окружения: `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD`.

Функции в `research.db.connection`: `build_postgres_url()`, `get_engine()`.
