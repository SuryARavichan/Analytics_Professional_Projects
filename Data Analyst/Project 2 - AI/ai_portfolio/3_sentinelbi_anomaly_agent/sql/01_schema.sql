/* SentinelBI — single fact table, daily grain x region x channel. */
CREATE TABLE IF NOT EXISTS daily_metrics (
    date     DATE NOT NULL,
    region   TEXT NOT NULL,
    channel  TEXT NOT NULL,
    revenue  REAL NOT NULL,
    orders   INTEGER NOT NULL,
    sessions INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dm_date ON daily_metrics(date);
