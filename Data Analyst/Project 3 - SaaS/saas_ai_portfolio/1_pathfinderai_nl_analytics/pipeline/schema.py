"""
Auto-introspects the SQLite database into a compact schema description
suitable for injecting into an LLM prompt (schema-aware prompting).
Includes column types, foreign-key-style relationships inferred from
naming convention, and 2 sample rows per table so the model sees real
value formats (e.g. that `status` is 'Completed'/'Returned'/'Cancelled',
not booleans) -- this measurably improves text-to-SQL accuracy versus
handing the model bare column names.
"""
import sqlite3


def describe_schema(db_path: str, sample_rows: int = 2) -> str:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]

    lines = []
    for t in tables:
        cur.execute(f"PRAGMA table_info({t})")
        cols = cur.fetchall()  # cid, name, type, notnull, dflt, pk
        col_desc = ", ".join(f"{c[1]} {c[2]}" for c in cols)
        lines.append(f"TABLE {t}({col_desc})")

        cur.execute(f"SELECT * FROM {t} LIMIT {sample_rows}")
        rows = cur.fetchall()
        col_names = [c[1] for c in cols]
        for r in rows:
            row_str = ", ".join(f"{n}={v!r}" for n, v in zip(col_names, r))
            lines.append(f"  sample row: {row_str}")
    conn.close()

    relationships = (
        "RELATIONSHIPS (inferred from foreign-key naming convention):\n"
        "  users.account_id -> accounts.account_id\n"
        "  subscriptions.account_id -> accounts.account_id  (one row per account -- current/most-recent subscription state)\n"
        "  product_events.account_id -> accounts.account_id\n"
        "  product_events.user_id -> users.user_id\n"
        "NOTES:\n"
        "  - subscriptions.status is one of 'Trial/Not Converted', 'Active', 'Churned'.\n"
        "  - subscriptions.mrr is 0/NULL-equivalent for accounts that never converted to paid.\n"
        "  - product_events.event_type is one of 'signup', 'activated', 'feature_used', 'converted_to_paid', 'cancelled'.\n"
        "  - product_events.feature_name is only populated for 'feature_used' events (the feature name) and\n"
        "    'converted_to_paid' events (the plan tier chosen) -- it is NULL for 'signup'/'activated'/'cancelled'.\n"
        "  - 'Activation' means an account has a product_events row with event_type='activated'.\n"
        "  - accounts.signup_date is when the account started its trial; not every account converts to paid.\n"
    )
    return "\n".join(lines) + "\n\n" + relationships


if __name__ == "__main__":
    print(describe_schema("../data/retail.db"))
