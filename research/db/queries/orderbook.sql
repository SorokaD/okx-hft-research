-- Orderbook L10 snapshot
-- Params: inst_id, ts_start (optional), ts_end (optional)
SELECT inst_id, snapshot_id, ts_event, ts_ingest,
  bid_px_01, bid_sz_01, bid_px_02, bid_sz_02, bid_px_03, bid_sz_03,
  bid_px_04, bid_sz_04, bid_px_05, bid_sz_05, bid_px_06, bid_sz_06,
  bid_px_07, bid_sz_07, bid_px_08, bid_sz_08, bid_px_09, bid_sz_09,
  bid_px_10, bid_sz_10,
  ask_px_01, ask_sz_01, ask_px_02, ask_sz_02, ask_px_03, ask_sz_03,
  ask_px_04, ask_sz_04, ask_px_05, ask_sz_05, ask_px_06, ask_sz_06,
  ask_px_07, ask_sz_07, ask_px_08, ask_sz_08, ask_px_09, ask_sz_09,
  ask_px_10, ask_sz_10,
  mid_px, spread_px
FROM okx_core.fact_orderbook_l10_snapshot
WHERE inst_id = :inst_id AND ts_event BETWEEN :ts_start AND :ts_end
ORDER BY ts_event;
