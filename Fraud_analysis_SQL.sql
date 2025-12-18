---Fraud_analysis_SQL


select * from dbo.fraudTrain;

select count(*) as total_records,
        SUM(is_fraud) as total_frauds,
        Round(AVG(amt),2) as avg_amount,
        MAX(amt) as max_amount,
        MIN(amt) as min_amount,
        MIN(trans_date_trans_time) as earliest_transaction,
        Max(trans_date_trans_time) as latest_transaction,
        Round(sum(is_fraud) * 1.0 / count(*) * 100,3) as fraud_percentage
 from dbo.fraudTrain;

--Daily Transaction and Fraud Trend

SELECT CAST(trans_date_trans_time AS DATE) AS transaction_date,
       COUNT(*) AS total_transactions,
       SUM(is_fraud) AS total_frauds,
       ROUND(SUM(is_fraud) * 1.0 / COUNT(*) * 100,3) AS fraud_percentage,
       ROUND(AVG(amt),2) AS avg_amount
FROM dbo.fraudTrain
GROUP BY CAST(trans_date_trans_time AS DATE)
ORDER BY transaction_date;

--Missing Value Check 

SELECT 
    SUM(case when cc_num is null then 1 else 0 end) as missing_cc_num,
    SUM(case when merchant is null then 1 else 0 end) as missing_merchant,
    SUM(case when category is null then 1 else 0 end) as missing_category,
    sum(case when amt is null then 1 else 0 end) as missing_null,
    sum(case when city is null then 1 else 0  end) as missing_city
from dbo.fraudTrain

--Transaction Amount Distribution

SELECT
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY amt) as p25,
    PERCENTILE_CONT(0.5) WITHIN GROUP(ORDER BY amt) as p50,
    PERCENTILE_CONT(0.75) WITHIN GROUP(ORDER BY amt) as p75,
    MAX(amt) as max_amt
from dbo.fraudTrain;


SELECT 
    CAST(trans_date_trans_time as DATE) as txn_date,
    COUNT(*) as total_txn,
    SUM(is_fraud) as fraud_txn,
    ROUND(SUM(is_fraud) * 100.0/ NULLIF(COUNT(*),0), 2) as fraud_rate_pct,
    ROUND(AVG(amt), 2) as avg_amount
FROM dbo.fraudTrain
GROUP BY CAST(trans_date_trans_time as DATE)
ORDER BY txn_date;

--High-risk customers with minimum sample thresholds

WITH per_customer AS (
    SELECT
        cc_num AS customer_id,
        COUNT(*) AS total_txn,
        SUM(is_fraud) AS fraud_txn,
        ROUND(AVG(CAST(is_fraud AS FLOAT)) * 100, 2) AS fraud_rate_pct,
        ROUND(SUM(amt), 2) AS total_spent,
        MIN(trans_date_trans_time) AS first_txn_ts,
        MAX(trans_date_trans_time) AS last_txn_ts
    FROM dbo.fraudTrain
    GROUP BY cc_num
)
SELECT TOP 20 *
FROM per_customer
WHERE total_txn >= 50
  AND fraud_txn >= 3
ORDER BY fraud_rate_pct DESC,
         fraud_txn DESC,
         total_spent DESC;

--Fraud vs Non-Fraud Behavior Comparison

SELECT
    is_fraud,
    gender,
    ROUND(AVG(CAST(amt AS FLOAT)), 2) AS avg_amount,
    ROUND(AVG(CAST(city_pop AS FLOAT)), 2) AS avg_city_pop,
    ROUND(AVG(CAST(DATEDIFF(YEAR, dob, '2020-06-21') AS FLOAT)), 1) AS avg_age
FROM dbo.fraudTrain
GROUP BY is_fraud, gender;

--Category-wise Fraud Analysis

SELECT
    Top 20
    category,
    COUNT(*)                    AS total_txn,
    SUM(is_fraud)               AS fraud_txn,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2) AS fraud_rate_pct,
    ROUND(AVG(amt), 2)          AS avg_amount
FROM dbo.fraudTrain
GROUP BY category
ORDER BY fraud_rate_pct DESC;

--Merchant-wise Fraud Analysis

SELECT
    Top 20 merchant,
    COUNT(*)                    AS total_txn,
    SUM(is_fraud)               AS fraud_txn,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2) AS fraud_rate_pct,
    ROUND(AVG(amt), 2)          AS avg_amount
FROM dbo.fraudTrain
GROUP BY merchant
HAVING COUNT(*) > 1000
ORDER BY fraud_rate_pct DESC;

--Hourly Fraud Trend Analysis

select 
    Top 20 merchant,
    DATEPART(HOUR, trans_date_trans_time) AS transaction_hour,
    COUNT(*) AS total_txn,  
    SUM(is_fraud) AS fraud_txn,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2) AS fraud_rate_pct,
    ROUND(AVG(amt), 2) AS avg_amount    
from dbo.fraudTrain
group by DATEPART(HOUR, trans_date_trans_time), merchant
order by fraud_rate_pct DESC;

--State-wise Fraud Analysis

SELECT
    Top 20
    state,
    COUNT(*)                    AS total_txn,
    SUM(is_fraud)               AS fraud_txn,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2) AS fraud_rate_pct,
    ROUND(AVG(amt), 2)          AS avg_amount                           
FROM dbo.fraudTrain
GROUP BY state;