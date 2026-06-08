

-- Step 1: Preview raw transaction data
SELECT *
FROM retail_transactions_day1
LIMIT 10;

--Step 2: Check raw row, invoice and customer counts
SELECT
   COUNT(*) AS total_rows,
   COUNT(DISTINCT invoice_no) AS total_invoices,
   COUNT(DISTINCT customer_id) AS total_customers
FROM retail_transactions_day1;

--Step 3: Check cancellation invoices
SELECT *
FROM retail_transactions_day1
WHERE invoice_no LIKE 'C%';

--Step 4:Check the invalid numbers
SELECT *
FROM retail_transactions_day1
WHERE quantity<=0;

--Step 5: Check the invalid price
SELECT *
FROM retail_transactions_day1
WHERE unit_price<=0;

--Step 6: Check the missing customer ID
SELECT *
FROM retail_transactions_day1
WHERE customer_id IS NULL;

--Step 7: Check the repaeated rows
SELECT
    invoice_no,
	stock_code,
	description,
	quantity,
	invoice_date,
	unit_price,
	customer_id,
	country,
	COUNT(*) AS duplicate_count
FROM retail_transactions_day1
GROUP BY 
      invoice_no,
	  stock_code,
	  description,
	  quantity,
	  invoice_date,
	  unit_price,
	  customer_id,
	  country
HAVING COUNT(*)>1;

--Step 8: Create a cleaned view of the retail transactions data
create or replace view clean_transactions as
select distinct
       invoice_no,
       stock_code,
       description,
       quantity,
       invoice_date,
       unit_price,
       customer_id,
       country,
       quantity * unit_price AS line_value
from retail_transactions_day1
where customer_id is not NULL
  and quantity>0
  and unit_price >0
  and invoice_no not like 'C%';

--Step 9: Clean before vs Clean after
select 
   'raw_transaction' as dataset,
   count(*) as transaction_lines,
   count(distinct invoice_no) as invoices,
   count(distinct customer_id) as customers,
   sum(quantity * unit_price) as revenue
from retail_transactions_day1

union all

select 
   'clean_transaction' as dataset,
   count(*) as transaction_lines,
   count(distinct invoice_no) as invoices,
   count(distinct customer_id) as customers,
   sum(quantity*unit_price) as revenue
from clean_transactions;   
   
   