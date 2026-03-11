"""
Orderbook L10 snapshot query.
Params: inst_id, optional ts_start, ts_end (timestamptz strings)
"""

ORDERBOOK_SNAPSHOT = """
SELECT
  inst_id, snapshot_id, ts_event, ts_ingest, ts_event_ms, ts_ingest_ms,
  bid_px_01, bid_sz_01, bid_px_02, bid_sz_02, bid_px_03, bid_sz_03,
  bid_px_04, bid_sz_04, bid_px_05, bid_sz_05, bid_px_06, bid_sz_06,
  bid_px_07, bid_sz_07, bid_px_08, bid_sz_08, bid_px_09, bid_sz_09,
  bid_px_10, bid_sz_10,
  ask_px_01, ask_sz_01, ask_px_02, ask_sz_02, ask_px_03, ask_sz_03,
  ask_px_04, ask_sz_04, ask_px_05, ask_sz_05, ask_px_06, ask_sz_06,
  ask_px_07, ask_sz_07, ask_px_08, ask_sz_08, ask_px_09, ask_sz_09,
  ask_px_10, ask_sz_10,
  mid_px, spread_px
FROM {schema}.{table}
WHERE inst_id = :inst_id
  AND (CAST(:ts_start AS timestamptz) IS NULL OR ts_event >= CAST(:ts_start AS timestamptz))
  AND (CAST(:ts_end AS timestamptz) IS NULL OR ts_event <= CAST(:ts_end AS timestamptz))
ORDER BY ts_event
"""
