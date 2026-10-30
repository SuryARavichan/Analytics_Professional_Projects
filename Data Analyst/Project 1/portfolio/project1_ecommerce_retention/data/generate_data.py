"""
Generates a realistic (synthetic) 2-year DTC e-commerce dataset for the
Customer Retention & Cohort Analysis project.

Story baked into the data (for analysts to discover):
  - Referral & Organic customers retain ~2x better than Paid Social customers.
  - A fulfillment / shipping-cost problem hit in Aug-Sep 2025, causing a
    visible retention dip in cohorts acquired right before and during that
    window (a realistic "diagnose the business problem" hook).
  - Home & Kitchen and Beauty categories have short repurchase cycles
    (high frequency, lower AOV); Electronics has long cycles (low frequency,
    high AOV) -- a realistic mixed-category retail assortment.

Output: CSV files in ./  and a SQLite DB (ecommerce.db) with the same tables,
so the project can be worked in SQL, pandas, Tableau/Power BI, or Excel.
"""
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta

rng = np.random.default_rng(42)

START = datetime(2024, 1, 1)
END = datetime(2025, 12, 31)
N_CUSTOMERS = 6000

CHANNELS = ["Organic Search", "Referral", "Paid Social", "Paid Search", "Email"]
CHANNEL_WEIGHTS = [0.22, 0.13, 0.30, 0.20, 0.15]
# Base monthly repeat-purchase probability per channel (drives retention story)
CHANNEL_RETENTION_MULT = {
    "Organic Search": 1.15,
    "Referral": 1.45,
    "Paid Social": 0.55,
    "Paid Search": 0.85,
    "Email": 1.10,
}

REGIONS = ["Northeast", "Midwest", "South", "West"]
REGION_WEIGHTS = [0.22, 0.21, 0.33, 0.24]

CATEGORIES = {
    "Home & Kitchen":  {"weight": 0.28, "price_range": (15, 90),  "repurchase_days": 35},
    "Beauty":          {"weight": 0.20, "price_range": (8, 60),   "repurchase_days": 28},
    "Apparel":         {"weight": 0.22, "price_range": (20, 120), "repurchase_days": 55},
    "Electronics":     {"weight": 0.12, "price_range": (40, 350), "repurchase_days": 120},
    "Sporting Goods":  {"weight": 0.10, "price_range": (15, 150), "repurchase_days": 70},
    "Pet Supplies":    {"weight": 0.08, "price_range": (10, 70),  "repurchase_days": 30},
}
CAT_NAMES = list(CATEGORIES.keys())
CAT_WEIGHTS = [CATEGORIES[c]["weight"] for c in CAT_NAMES]

PAYMENT_METHODS = ["Credit Card", "Debit Card", "PayPal", "Buy Now Pay Later"]
PAYMENT_WEIGHTS = [0.48, 0.22, 0.20, 0.10]


def random_dates(start, end, n):
    delta_days = (end - start).days
    offsets = rng.integers(0, delta_days + 1, size=n)
    return [start + timedelta(days=int(o)) for o in offsets]


# ---------------------------------------------------------------------------
# 1. CUSTOMERS
# ---------------------------------------------------------------------------
customer_ids = [f"CUST{100000+i}" for i in range(N_CUSTOMERS)]
# Signups spread across the window, slightly increasing volume over time
month_index = np.arange(24)
signup_weights = np.clip(1 + 0.03 * month_index + rng.normal(0, 0.15, 24), 0.2, None)
signup_weights = signup_weights / signup_weights.sum()
signup_months = rng.choice(month_index, size=N_CUSTOMERS, p=signup_weights)

signup_dates = []
for m in signup_months:
    month_start = START + pd.DateOffset(months=int(m))
    days_in_month = (month_start + pd.DateOffset(months=1) - month_start).days
    d = rng.integers(0, days_in_month)
    signup_dates.append((month_start + timedelta(days=int(d))).to_pydatetime())

channels = rng.choice(CHANNELS, size=N_CUSTOMERS, p=CHANNEL_WEIGHTS)
regions = rng.choice(REGIONS, size=N_CUSTOMERS, p=REGION_WEIGHTS)
age_brackets = rng.choice(["18-24", "25-34", "35-44", "45-54", "55+"], size=N_CUSTOMERS,
                           p=[0.14, 0.32, 0.26, 0.16, 0.12])

customers = pd.DataFrame({
    "customer_id": customer_ids,
    "signup_date": pd.to_datetime(signup_dates).date,
    "acquisition_channel": channels,
    "region": regions,
    "age_bracket": age_brackets,
})

# ---------------------------------------------------------------------------
# 2. ORDERS / ORDER_ITEMS / PAYMENTS / REVIEWS
# ---------------------------------------------------------------------------
orders_rows, items_rows, payments_rows, reviews_rows = [], [], [], []
order_counter = 1
item_counter = 1

for _, cust in customers.iterrows():
    signup_dt = pd.Timestamp(cust["signup_date"])
    channel = cust["acquisition_channel"]
    mult = CHANNEL_RETENTION_MULT[channel]

    # First (acquisition) order happens within 0-3 days of signup
    first_order_date = signup_dt + timedelta(days=int(rng.integers(0, 4)))
    if first_order_date > pd.Timestamp(END):
        continue

    current_date = first_order_date
    n_orders_for_customer = 0
    while current_date <= pd.Timestamp(END):
        n_orders_for_customer += 1
        order_id = f"ORD{200000+order_counter}"
        order_counter += 1

        # 1-4 items per order
        n_items = rng.choice([1, 2, 3, 4], p=[0.45, 0.30, 0.16, 0.09])
        order_value = 0
        for _ in range(n_items):
            cat = rng.choice(CAT_NAMES, p=CAT_WEIGHTS)
            lo, hi = CATEGORIES[cat]["price_range"]
            unit_price = round(rng.uniform(lo, hi), 2)
            qty = rng.choice([1, 2, 3], p=[0.75, 0.18, 0.07])
            items_rows.append({
                "order_item_id": f"ITEM{300000+item_counter}",
                "order_id": order_id,
                "product_category": cat,
                "quantity": int(qty),
                "unit_price": unit_price,
            })
            order_value += unit_price * qty
            item_counter += 1

        # order status: mostly delivered, some cancelled/returned
        status = rng.choice(["Delivered", "Returned", "Cancelled"], p=[0.90, 0.07, 0.03])

        # --- The Aug-Sep 2025 fulfillment problem: elevated cancel/return + drop in reorder rate ---
        in_problem_window = pd.Timestamp("2025-08-01") <= current_date <= pd.Timestamp("2025-09-30")
        if in_problem_window:
            status = rng.choice(["Delivered", "Returned", "Cancelled"], p=[0.78, 0.14, 0.08])

        orders_rows.append({
            "order_id": order_id,
            "customer_id": cust["customer_id"],
            "order_date": current_date.date(),
            "acquisition_channel_at_order": channel,
            "order_status": status,
            "order_value": round(order_value, 2),
        })

        payment_method = rng.choice(PAYMENT_METHODS, p=PAYMENT_WEIGHTS)
        payments_rows.append({
            "order_id": order_id,
            "payment_method": payment_method,
            "payment_value": round(order_value, 2),
        })

        if status != "Cancelled" and rng.random() < 0.72:
            base_score = 4.3
            if in_problem_window:
                base_score -= 0.9
            score = int(np.clip(round(rng.normal(base_score, 0.9)), 1, 5))
            reviews_rows.append({
                "order_id": order_id,
                "review_score": score,
                "review_date": (current_date + timedelta(days=int(rng.integers(2, 10)))).date(),
            })

        # decide repurchase gap using dominant category cycle length, channel multiplier,
        # and a global penalty if this order fell in the fulfillment-problem window
        cats_in_order = [r["product_category"] for r in items_rows[-n_items:]]
        avg_cycle = np.mean([CATEGORIES[c]["repurchase_days"] for c in cats_in_order])
        problem_penalty = 1.6 if in_problem_window else 1.0
        gap_days = rng.exponential(avg_cycle / mult * problem_penalty) + 5

        # churn check: probability customer never returns grows with each order already placed
        churn_chance = 0.12 + 0.02 * n_orders_for_customer
        if in_problem_window:
            churn_chance += 0.15
        if rng.random() < churn_chance and n_orders_for_customer > 1:
            break

        current_date = current_date + timedelta(days=float(gap_days))

orders = pd.DataFrame(orders_rows)
order_items = pd.DataFrame(items_rows)
payments = pd.DataFrame(payments_rows)
reviews = pd.DataFrame(reviews_rows)

print(f"customers={len(customers)}  orders={len(orders)}  order_items={len(order_items)}  "
      f"payments={len(payments)}  reviews={len(reviews)}")

# ---------------------------------------------------------------------------
# Save CSVs
# ---------------------------------------------------------------------------
customers.to_csv("customers.csv", index=False)
orders.to_csv("orders.csv", index=False)
order_items.to_csv("order_items.csv", index=False)
payments.to_csv("payments.csv", index=False)
reviews.to_csv("reviews.csv", index=False)

# ---------------------------------------------------------------------------
# Save SQLite DB
# ---------------------------------------------------------------------------
conn = sqlite3.connect("ecommerce.db")
customers.to_sql("customers", conn, if_exists="replace", index=False)
orders.to_sql("orders", conn, if_exists="replace", index=False)
order_items.to_sql("order_items", conn, if_exists="replace", index=False)
payments.to_sql("payments", conn, if_exists="replace", index=False)
reviews.to_sql("reviews", conn, if_exists="replace", index=False)
conn.close()

print("Done. Wrote CSVs + ecommerce.db")
