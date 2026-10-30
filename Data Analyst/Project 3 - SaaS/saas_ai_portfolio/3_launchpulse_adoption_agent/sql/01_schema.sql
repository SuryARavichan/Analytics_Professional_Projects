/* LaunchPulse — two fact tables.
   daily_metrics: feature-adoption events at date x feature x channel x plan grain.
   daily_channel_funnel: signup->activation funnel at date x channel grain. */
CREATE TABLE IF NOT EXISTS daily_metrics (
    date            DATE NOT NULL,
    feature_name    TEXT NOT NULL,
    channel         TEXT NOT NULL,
    plan_tier       TEXT NOT NULL,
    adoption_events INTEGER NOT NULL,
    active_accounts INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dm_date ON daily_metrics(date);

CREATE TABLE IF NOT EXISTS daily_channel_funnel (
    date        DATE NOT NULL,
    channel     TEXT NOT NULL,
    signups     INTEGER NOT NULL,
    activations INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dcf_date ON daily_channel_funnel(date);
