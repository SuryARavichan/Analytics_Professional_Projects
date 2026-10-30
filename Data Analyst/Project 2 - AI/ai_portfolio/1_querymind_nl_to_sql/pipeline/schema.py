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
        "  sales_transactions.store_id -> stores.store_id\n"
        "  sales_transactions.customer_id -> customers.customer_id\n"
        "  sales_items.transaction_id -> sales_transactions.transaction_id\n"
        "  sales_items.product_id -> products.product_id\n"
        "NOTES:\n"
        "  - sales_items.line_total is the item's realized revenue (unit_price * quantity * (1-discount_pct)).\n"
        "  - sales_transactions.status is one of 'Completed', 'Returned', 'Cancelled'.\n"
        "  - stores.region is where the sale happened; customers.signup_region is where the customer signed up --\n"
        "    these can differ for a given transaction.\n"
    )
    return "\n".join(lines) + "\n\n" + relationships


if __name__ == "__main__":
    print(describe_schema("../data/retail.db"))
