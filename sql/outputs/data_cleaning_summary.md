# Data Cleaning Summary

| Dataset | Transaction Lines | Invoices | Customers | Revenue |
|---|---:|---:|---:|---:|
| Raw transactions | 34 | 30 | 24 | 1256.50 |
| Clean transactions | 29 | 26 | 22 | 1216.00 |

## Cleaning Rules

- Removed cancellation invoices where invoice_no starts with C.
- Removed rows without customer_id.
- Removed rows with quantity <= 0.
- Removed rows with unit_price <= 0.
- Removed fully duplicated transaction lines.
