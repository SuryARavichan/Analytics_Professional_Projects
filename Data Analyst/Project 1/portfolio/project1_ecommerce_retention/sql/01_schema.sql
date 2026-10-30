/* ============================================================
   Project 1 — E-commerce Customer Retention & Cohort Analysis
   Schema reference (SQLite; the same schema works in Postgres/MySQL
   with minor type changes — DATE/TEXT instead of SQLite's dynamic typing).
   ============================================================ */

CREATE TABLE IF NOT EXISTS customers (
    customer_id             TEXT PRIMARY KEY,
    signup_date             DATE NOT NULL,
    acquisition_channel     TEXT NOT NULL,
    region                  TEXT NOT NULL,
    age_bracket             TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    order_id                        TEXT PRIMARY KEY,
    customer_id                     TEXT NOT NULL REFERENCES customers(customer_id),
    order_date                      DATE NOT NULL,
    acquisition_channel_at_order    TEXT NOT NULL,
    order_status                    TEXT NOT NULL CHECK (order_status IN ('Delivered','Returned','Cancelled')),
    order_value                     REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS order_items (
    order_item_id       TEXT PRIMARY KEY,
    order_id             TEXT NOT NULL REFERENCES orders(order_id),
    product_category     TEXT NOT NULL,
    quantity              INTEGER NOT NULL,
    unit_price            REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS payments (
    order_id        TEXT NOT NULL REFERENCES orders(order_id),
    payment_method  TEXT NOT NULL,
    payment_value   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS reviews (
    order_id       TEXT NOT NULL REFERENCES orders(order_id),
    review_score   INTEGER NOT NULL CHECK (review_score BETWEEN 1 AND 5),
    review_date    DATE NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_date ON orders(order_date);
CREATE INDEX IF NOT EXISTS idx_items_order ON order_items(order_id);
