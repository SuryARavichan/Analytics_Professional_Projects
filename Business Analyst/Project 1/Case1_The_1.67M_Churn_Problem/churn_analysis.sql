-- =====================================================================
-- CASE 01 — CHURN INTELLIGENCE & RETENTION CASE
-- Dataset: IBM Telco Customer Churn (7,043 customers)
-- Source : https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d
-- Engine : SQLite3 (table: customers)
-- =====================================================================

-- 1. Overall churn rate
SELECT
  COUNT(*)                                            AS total_customers,
  SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)      AS churned,
  ROUND(100.0 * SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers;

-- 2. Churn rate by contract type
SELECT
  Contract,
  COUNT(*)                                            AS customers,
  SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END)        AS churned,
  ROUND(100.0 * SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;

-- 3. Churn rate by tenure bucket
SELECT
  CASE
    WHEN tenure <= 12 THEN '0-12 mo'
    WHEN tenure <= 24 THEN '13-24 mo'
    WHEN tenure <= 48 THEN '25-48 mo'
    WHEN tenure <= 60 THEN '49-60 mo'
    ELSE '61-72 mo'
  END AS tenure_bucket,
  COUNT(*)                                            AS customers,
  ROUND(100.0 * SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY tenure_bucket
ORDER BY MIN(tenure);

-- 4. Churn rate by payment method
SELECT
  PaymentMethod,
  COUNT(*)                                            AS customers,
  ROUND(100.0 * SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;

-- 5. Churn rate by internet service type
SELECT
  InternetService,
  COUNT(*)                                            AS customers,
  ROUND(100.0 * SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY InternetService
ORDER BY churn_rate_pct DESC;

-- 6. Does a support/security bundle reduce churn? (Fiber customers only)
SELECT
  CASE WHEN OnlineSecurity='No' AND TechSupport='No' THEN 'No security + no support'
       ELSE 'Has security and/or support' END          AS bundle_status,
  COUNT(*)                                             AS customers,
  ROUND(100.0 * SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers
WHERE InternetService = 'Fiber optic'
GROUP BY bundle_status;

-- 7. Average monthly charges: churned vs retained
SELECT
  Churn,
  ROUND(AVG(MonthlyCharges), 2)                        AS avg_monthly_charges,
  COUNT(*)                                             AS customers
FROM customers
GROUP BY Churn;

-- 8. Monthly revenue currently at risk (churned customers' monthly charges)
SELECT
  ROUND(SUM(MonthlyCharges), 2)                        AS monthly_revenue_lost,
  ROUND(SUM(MonthlyCharges) * 12, 2)                   AS annualized_revenue_lost
FROM customers
WHERE Churn = 'Yes';

-- 9. Highest-risk segment: month-to-month + fiber + no tech support
SELECT
  COUNT(*)                                             AS customers_in_segment,
  ROUND(100.0 * SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
  ROUND(SUM(CASE WHEN Churn='Yes' THEN MonthlyCharges ELSE 0 END), 2)       AS monthly_revenue_at_risk
FROM customers
WHERE Contract = 'Month-to-month'
  AND InternetService = 'Fiber optic'
  AND TechSupport = 'No';

-- 10. Paperless billing x payment method interaction (electronic check risk)
SELECT
  PaymentMethod,
  PaperlessBilling,
  COUNT(*)                                             AS customers,
  ROUND(100.0 * SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY PaymentMethod, PaperlessBilling
ORDER BY churn_rate_pct DESC
LIMIT 5;
