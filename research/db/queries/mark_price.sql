-- Mark price
-- Params: inst_id, ts_start (optional), ts_end (optional)
SELECT inst_id, ts_event, ts_ingest, mark_px
FROM okx_core.fact_mark_price_tick
WHERE inst_id = :inst_id AND ts_event BETWEEN :ts_start AND :ts_end
ORDER BY ts_event;
