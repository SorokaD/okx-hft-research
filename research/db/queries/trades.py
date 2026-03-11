"""
Trades query.
Params: inst_id, optional ts_start, ts_end (timestamptz strings)
"""

TRADES = """
SELECT
  inst_id, trade_id, ts_event, ts_ingest, price, size, side
FROM {schema}.{table}
WHERE inst_id = :inst_id
  AND (CAST(:ts_start AS timestamptz) IS NULL OR ts_event >= CAST(:ts_start AS timestamptz))
  AND (CAST(:ts_end AS timestamptz) IS NULL OR ts_event <= CAST(:ts_end AS timestamptz))
ORDER BY ts_event
"""
