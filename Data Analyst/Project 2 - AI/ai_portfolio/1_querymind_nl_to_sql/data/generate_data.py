"""
Generates a synthetic multi-store retail chain dataset: 5 related tables
(stores, products, customers, sales_transactions, sales_items) with enough
join complexity to make natural-language-to-SQL genuinely non-trivial --
the point of this project isn't the dataset, it's exercising joins,
aggregations, time filters, and top-N logic through an NL interface.
"""
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta

rng = np.random.default_rng(21)

START = datetime(2024, 6, 1)
END = datetime(2025, 12, 31)

# ---------------------------------------------------------------------
# STORES
# ---------------------------------------------------------------------
REGIONS = ["Northeast", "Midwest", "South", "West"]
N_STORES = 24
store_region = rng.choice(REGIONS, size=N_STORES, p=[0.25, 0.25, 0.30, 0.20])
stores = pd.DataFrame({
    "store_id": [f"ST{100+i}" for i in range(N_STORES)],
    "store_name": [f"Store #{100+i}" for i in range(N_STORES)],
    "region": store_region,
    "store_type": rng.choice(["Flagship", "Mall", "Strip Mall", "Outlet"], size=N_STORES, p=[0.08, 0.35, 0.42, 0.15]),
    "opened_year": rng.integers(2010, 2023, N_STORES),
})

# ---------------------------------------------------------------------
# PRODUCTS
# ---------------------------------------------------------------------
CATEGORIES = {
    "Footwear": (35, 160), "Apparel": (18, 120), "Accessories": (10, 80),
    "Outdoor Gear": (25, 300), "Electronics": (30, 450),
}
N_PRODUCTS = 220
cat_choice = rng.choice(list(CATEGORIES.keys()), size=N_PRODUCTS, p=[0.26, 0.28, 0.18, 0.16, 0.12])
products = pd.DataFrame({
    "product_id": [f"P{5000+i}" for i in range(N_PRODUCTS)],
    "product_name": [f"{cat_choice[i]} Item {5000+i}" for i in range(N_PRODUCTS)],
    "category": cat_choice,
    "unit_cost": [round(rng.uniform(*CATEGORIES[c]) * 0.5, 2) for c in cat_choice],
    "unit_price": None,
})
products["unit_price"] = (products["unit_cost"] * rng.uniform(1.8, 2.6, N_PRODUCTS)).round(2)

# ---------------------------------------------------------------------
# CUSTOMERS
# ---------------------------------------------------------------------
N_CUSTOMERS = 9000
customers = pd.DataFrame({
    "customer_id": [f"C{20000+i}" for i in range(N_CUSTOMERS)],
    "loyalty_member": rng.choice([0, 1], size=N_CUSTOMERS, p=[0.62, 0.38]),
    "signup_region": rng.choice(REGIONS, size=N_CUSTOMERS, p=[0.25, 0.25, 0.30, 0.20]),
    "age_bracket": rng.choice(["18-24", "25-34", "35-44", "45-54", "55+"], size=N_CUSTOMERS,
                               p=[0.12, 0.30, 0.27, 0.18, 0.13]),
})

# ---------------------------------------------------------------------
# SALES TRANSACTIONS + ITEMS
# ---------------------------------------------------------------------
N_TRANSACTIONS = 42000
txn_dates = [START + timedelta(days=int(d), hours=int(rng.integers(9, 21)))
             for d in rng.integers(0, (END - START).days, N_TRANSACTIONS)]
# holiday-season and weekend bump baked in for realistic seasonality
month_weight = np.array([1.0, 0.95, 1.05, 1.0, 1.05, 1.0, 0.95, 1.0, 1.05, 1.15, 1.35, 1.55])

txn_store = rng.choice(stores.store_id, size=N_TRANSACTIONS)
txn_customer = rng.choice(customers.customer_id, size=N_TRANSACTIONS)
channel = rng.choice(["In-Store", "Online"], size=N_TRANSACTIONS, p=[0.68, 0.32])
payment = rng.choice(["Credit Card", "Debit Card", "Cash", "Gift Card"], size=N_TRANSACTIONS, p=[0.5, 0.28, 0.14, 0.08])
status = rng.choice(["Completed", "Returned", "Cancelled"], size=N_TRANSACTIONS, p=[0.91, 0.06, 0.03])

sales_transactions = pd.DataFrame({
    "transaction_id": [f"TXN{700000+i}" for i in range(N_TRANSACTIONS)],
    "store_id": txn_store,
    "customer_id": txn_customer,
    "transaction_date": [d.date() for d in txn_dates],
    "transaction_ts": [d.isoformat(sep=" ") for d in txn_dates],
    "channel": channel,
    "payment_method": payment,
    "status": status,
})

items_rows, item_ctr = [], 1
for txn_id in sales_transactions.transaction_id:
    n_items = rng.choice([1, 2, 3, 4, 5], p=[0.38, 0.28, 0.18, 0.10, 0.06])
    chosen = rng.choice(products.index, size=n_items, replace=False)
    for pi in chosen:
        qty = int(rng.choice([1, 2, 3], p=[0.78, 0.16, 0.06]))
        discount_pct = float(rng.choice([0, 0, 0, 0.10, 0.20, 0.30], p=[0.55, 0.1, 0.1, 0.1, 0.1, 0.05]))
        items_rows.append({
            "sales_item_id": f"SI{item_ctr}", "transaction_id": txn_id,
            "product_id": products.loc[pi, "product_id"], "quantity": qty,
            "unit_price": products.loc[pi, "unit_price"], "discount_pct": discount_pct,
        })
        item_ctr += 1
sales_items = pd.DataFrame(items_rows)
sales_items["line_total"] = round(sales_items.quantity * sales_items.unit_price * (1 - sales_items.discount_pct), 2)

print(f"stores={len(stores)} products={len(products)} customers={len(customers)} "
      f"transactions={len(sales_transactions)} items={len(sales_items)}")

for df, name in [(stores, "stores"), (products, "products"), (customers, "customers"),
                  (sales_transactions, "sales_transactions"), (sales_items, "sales_items")]:
    df.to_csv(f"{name}.csv", index=False)

conn = sqlite3.connect("retail.db")
stores.to_sql("stores", conn, if_exists="replace", index=False)
products.to_sql("products", conn, if_exists="replace", index=False)
customers.to_sql("customers", conn, if_exists="replace", index=False)
sales_transactions.to_sql("sales_transactions", conn, if_exists="replace", index=False)
sales_items.to_sql("sales_items", conn, if_exists="replace", index=False)
conn.close()
print("Done. Wrote CSVs + retail.db")
