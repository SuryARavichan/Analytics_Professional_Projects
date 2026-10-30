"""
Defines the 20-question QueryMind evaluation benchmark: each question has
a hand-verified GOLD SQL query (the correct answer) and a PREDICTED SQL
query representing what the LLM produced in this cached demo run. 16 of
20 predictions are correct; 4 contain a deliberately realistic text-to-SQL
failure mode (grain/denominator error, missing DISTINCT causing join
fan-out, a dropped status filter, and an ambiguous-column interpretation)
-- the same failure classes that show up in real LLM-to-SQL systems, so
the eval below is demonstrating something real, not a rigged 100%.

Running this script (re)builds:
  - benchmark.json               the 20 question/gold/predicted/note records
  - cached_sql_responses.json    prompt_id -> predicted SQL (feeds llm_client's
                                  CachedClient so text_to_sql.py's code path is
                                  identical whether the SQL came from a demo
                                  fixture or a live model)
"""
import json

BENCH = [
    dict(id="q01", difficulty="easy",
         question="What was total sales revenue in Q4 2025 (Oct-Dec)?",
         gold="""SELECT ROUND(SUM(si.line_total),2) AS total_revenue
                 FROM sales_items si JOIN sales_transactions t ON t.transaction_id=si.transaction_id
                 WHERE t.status='Completed' AND t.transaction_date BETWEEN '2025-10-01' AND '2025-12-31';""",
         predicted=None, note=None),

    dict(id="q02", difficulty="medium",
         question="Which store had the highest total revenue in 2025?",
         gold="""SELECT s.store_name, ROUND(SUM(si.line_total),2) AS revenue
                 FROM sales_items si JOIN sales_transactions t ON t.transaction_id=si.transaction_id
                 JOIN stores s ON s.store_id=t.store_id
                 WHERE t.status='Completed' AND t.transaction_date BETWEEN '2025-01-01' AND '2025-12-31'
                 GROUP BY s.store_id ORDER BY revenue DESC LIMIT 1;""",
         predicted=None, note=None),

    dict(id="q03", difficulty="medium",
         question="What is the average transaction value by channel?",
         gold="""WITH txn_totals AS (
                     SELECT t.transaction_id, t.channel, SUM(si.line_total) AS txn_total
                     FROM sales_transactions t JOIN sales_items si ON si.transaction_id=t.transaction_id
                     WHERE t.status='Completed' GROUP BY t.transaction_id)
                 SELECT channel, ROUND(AVG(txn_total),2) AS avg_transaction_value FROM txn_totals GROUP BY channel;""",
         predicted=None, note=None),

    dict(id="q04", difficulty="easy",
         question="How many loyalty members made a purchase in November 2025?",
         gold="""SELECT COUNT(DISTINCT t.customer_id)
                 FROM sales_transactions t JOIN customers c ON c.customer_id=t.customer_id
                 WHERE c.loyalty_member=1 AND t.status='Completed'
                   AND t.transaction_date BETWEEN '2025-11-01' AND '2025-11-30';""",
         predicted=None, note=None),

    dict(id="q05", difficulty="medium",
         question="What are the top 5 best-selling products by quantity sold?",
         gold="""SELECT p.product_name, SUM(si.quantity) AS total_qty
                 FROM sales_items si JOIN products p ON p.product_id=si.product_id
                 JOIN sales_transactions t ON t.transaction_id=si.transaction_id
                 WHERE t.status='Completed' GROUP BY p.product_id ORDER BY total_qty DESC LIMIT 5;""",
         predicted=None, note=None),

    dict(id="q06", difficulty="easy",
         question="Which product category generates the most revenue?",
         gold="""SELECT p.category, ROUND(SUM(si.line_total),2) AS revenue
                 FROM sales_items si JOIN products p ON p.product_id=si.product_id
                 JOIN sales_transactions t ON t.transaction_id=si.transaction_id
                 WHERE t.status='Completed' GROUP BY p.category ORDER BY revenue DESC LIMIT 1;""",
         predicted=None, note=None),

    dict(id="q07", difficulty="hard",
         question="What percentage of transactions were returned?",
         gold="""SELECT ROUND(100.0*SUM(CASE WHEN status='Returned' THEN 1 ELSE 0 END)/COUNT(*),2) AS return_rate_pct
                 FROM sales_transactions;""",
         predicted="""SELECT ROUND(100.0*SUM(CASE WHEN t.status='Returned' THEN 1 ELSE 0 END)/COUNT(*),2) AS return_rate_pct
                 FROM sales_transactions t JOIN sales_items si ON si.transaction_id=t.transaction_id;""",
         note="Grain/denominator error: joining sales_items changes the row grain from one-per-transaction "
              "to one-per-line-item, so the denominator (and the miscounted numerator) is inflated by the "
              "average items-per-transaction -- the percentage is systematically wrong even though the SQL runs cleanly."),

    dict(id="q08", difficulty="medium",
         question="What is the average discount percentage applied on transactions in the West region?",
         gold="""SELECT ROUND(AVG(si.discount_pct)*100,2) AS avg_discount_pct
                 FROM sales_items si JOIN sales_transactions t ON t.transaction_id=si.transaction_id
                 JOIN stores s ON s.store_id=t.store_id WHERE s.region='West';""",
         predicted=None, note=None),

    dict(id="q09", difficulty="hard",
         question="How many unique customers purchased Electronics in 2025?",
         gold="""SELECT COUNT(DISTINCT t.customer_id)
                 FROM sales_transactions t JOIN sales_items si ON si.transaction_id=t.transaction_id
                 JOIN products p ON p.product_id=si.product_id
                 WHERE p.category='Electronics' AND t.status='Completed'
                   AND t.transaction_date BETWEEN '2025-01-01' AND '2025-12-31';""",
         predicted="""SELECT COUNT(t.customer_id)
                 FROM sales_transactions t JOIN sales_items si ON si.transaction_id=t.transaction_id
                 JOIN products p ON p.product_id=si.product_id
                 WHERE p.category='Electronics' AND t.status='Completed'
                   AND t.transaction_date BETWEEN '2025-01-01' AND '2025-12-31';""",
         note="Missing DISTINCT: joining sales_items fans out one row per Electronics line item, so a customer "
              "who bought 3 Electronics items across 2 transactions is counted 3 times instead of once."),

    dict(id="q10", difficulty="medium",
         question="What is the month-over-month revenue trend for 2025?",
         gold="""SELECT strftime('%Y-%m', t.transaction_date) AS month, ROUND(SUM(si.line_total),2) AS revenue
                 FROM sales_transactions t JOIN sales_items si ON si.transaction_id=t.transaction_id
                 WHERE t.status='Completed' AND t.transaction_date BETWEEN '2025-01-01' AND '2025-12-31'
                 GROUP BY month ORDER BY month;""",
         predicted="""SELECT strftime('%Y-%m', t.transaction_date) AS month, ROUND(SUM(si.line_total),2) AS revenue
                 FROM sales_transactions t JOIN sales_items si ON si.transaction_id=t.transaction_id
                 WHERE t.transaction_date BETWEEN '2025-01-01' AND '2025-12-31'
                 GROUP BY month ORDER BY month;""",
         note="Dropped the status='Completed' filter, so Returned and Cancelled transactions' revenue is "
              "incorrectly included in every month's total -- a subtle overstatement, not an obvious crash."),

    dict(id="q11", difficulty="hard",
         question="Which region has the highest average transaction value?",
         gold="""WITH txn_totals AS (
                     SELECT t.transaction_id, s.region, SUM(si.line_total) AS txn_total
                     FROM sales_transactions t JOIN sales_items si ON si.transaction_id=t.transaction_id
                     JOIN stores s ON s.store_id=t.store_id WHERE t.status='Completed' GROUP BY t.transaction_id)
                 SELECT region, ROUND(AVG(txn_total),2) AS avg_txn_value FROM txn_totals
                 GROUP BY region ORDER BY avg_txn_value DESC LIMIT 1;""",
         predicted=None, note=None),

    dict(id="q12", difficulty="medium",
         question="What's the return rate by store type?",
         gold="""SELECT s.store_type, ROUND(100.0*SUM(CASE WHEN t.status='Returned' THEN 1 ELSE 0 END)/COUNT(*),2) AS return_rate_pct
                 FROM sales_transactions t JOIN stores s ON s.store_id=t.store_id
                 GROUP BY s.store_type ORDER BY return_rate_pct DESC;""",
         predicted=None, note=None),

    dict(id="q13", difficulty="easy",
         question="How many transactions used a Gift Card as the payment method?",
         gold="""SELECT COUNT(*) FROM sales_transactions WHERE payment_method='Gift Card';""",
         predicted=None, note=None),

    dict(id="q14", difficulty="medium",
         question="What is total revenue from customers aged 25-34?",
         gold="""SELECT ROUND(SUM(si.line_total),2) AS revenue
                 FROM sales_items si JOIN sales_transactions t ON t.transaction_id=si.transaction_id
                 JOIN customers c ON c.customer_id=t.customer_id
                 WHERE c.age_bracket='25-34' AND t.status='Completed';""",
         predicted=None, note=None),

    dict(id="q15", difficulty="medium",
         question="What is the average number of items per transaction?",
         gold="""WITH item_counts AS (SELECT transaction_id, SUM(quantity) AS n_items FROM sales_items GROUP BY transaction_id)
                 SELECT ROUND(AVG(n_items),2) AS avg_items_per_txn FROM item_counts;""",
         predicted=None, note=None),

    dict(id="q16", difficulty="hard",
         question="What is the online vs in-store revenue split for the Northeast region?",
         gold="""SELECT t.channel, ROUND(SUM(si.line_total),2) AS revenue
                 FROM sales_transactions t JOIN sales_items si ON si.transaction_id=t.transaction_id
                 JOIN stores s ON s.store_id=t.store_id
                 WHERE s.region='Northeast' AND t.status='Completed' GROUP BY t.channel;""",
         predicted="""SELECT t.channel, ROUND(SUM(si.line_total),2) AS revenue
                 FROM sales_transactions t JOIN sales_items si ON si.transaction_id=t.transaction_id
                 JOIN customers c ON c.customer_id=t.customer_id
                 WHERE c.signup_region='Northeast' AND t.status='Completed' GROUP BY t.channel;""",
         note="Ambiguous-entity resolution: 'Northeast region' most naturally means the store/point-of-sale "
              "region, but the model resolved it to the customer's signup region instead -- a different, "
              "silently-plausible interpretation. Flags why NL-to-SQL output should surface its column choice "
              "for ambiguous business terms rather than being trusted blindly."),

    dict(id="q17", difficulty="medium",
         question="List the top 3 products by revenue in the Footwear category.",
         gold="""SELECT p.product_name, ROUND(SUM(si.line_total),2) AS revenue
                 FROM sales_items si JOIN products p ON p.product_id=si.product_id
                 JOIN sales_transactions t ON t.transaction_id=si.transaction_id
                 WHERE p.category='Footwear' AND t.status='Completed'
                 GROUP BY p.product_id ORDER BY revenue DESC LIMIT 3;""",
         predicted=None, note=None),

    dict(id="q18", difficulty="easy",
         question="How many Flagship stores are in each region?",
         gold="""SELECT region, COUNT(*) AS n_stores FROM stores WHERE store_type='Flagship' GROUP BY region ORDER BY region;""",
         predicted=None, note=None),

    dict(id="q19", difficulty="hard",
         question="What is the loyalty membership rate among customers who made a purchase in the South region?",
         gold="""WITH south_customers AS (
                     SELECT DISTINCT t.customer_id FROM sales_transactions t JOIN stores s ON s.store_id=t.store_id
                     WHERE s.region='South' AND t.status='Completed')
                 SELECT ROUND(100.0*SUM(c.loyalty_member)/COUNT(*),2) AS loyalty_rate_pct
                 FROM south_customers sc JOIN customers c ON c.customer_id=sc.customer_id;""",
         predicted=None, note=None),

    dict(id="q20", difficulty="easy",
         question="What's total revenue by payment method, from highest to lowest?",
         gold="""SELECT t.payment_method, ROUND(SUM(si.line_total),2) AS revenue
                 FROM sales_transactions t JOIN sales_items si ON si.transaction_id=t.transaction_id
                 WHERE t.status='Completed' GROUP BY t.payment_method ORDER BY revenue DESC;""",
         predicted=None, note=None),
]

for row in BENCH:
    if row["predicted"] is None:
        row["predicted"] = row["gold"]

with open("benchmark.json", "w") as f:
    json.dump(BENCH, f, indent=2)

cached_sql = {row["id"]: row["predicted"] for row in BENCH}
with open("cached_sql_responses.json", "w") as f:
    json.dump(cached_sql, f, indent=2)

print(f"Wrote benchmark.json ({len(BENCH)} questions) and cached_sql_responses.json")
print(f"Injected failures: {[r['id'] for r in BENCH if r['note']]}")
