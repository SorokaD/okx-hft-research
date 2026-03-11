"""
Mark price query.
Params: inst_id, optional ts_start, ts_end (timestamptz strings)
"""

MARK_PRICE = """
SELECT
  inst_id, ts_event, ts_ingest, mark_px
FROM {schema}.{table}
WHERE inst_id = :inst_id
  AND (:ts_start::timestamptz IS NULL OR ts_event >= :ts_start::timestamptz)
  AND (:ts_end::timestamptz IS NULL OR ts_event <= :ts_end::timestamptz)
ORDER BY ts_event
"""
