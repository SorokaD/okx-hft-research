-- Trades
-- Params: inst_id, ts_start (optional), ts_end (optional)
SELECT inst_id, trade_id, ts_event, ts_ingest, price, size, side
FROM okx_core.fact_trades_tick
WHERE inst_id = :inst_id AND ts_event BETWEEN :ts_start AND :ts_end
ORDER BY ts_event;
