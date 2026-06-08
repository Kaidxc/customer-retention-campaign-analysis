# Data Cleaning Summary

| Dataset | Transaction Lines | Invoices | Customers | Revenue |
|---|---:|---:|---:|---:|
| Raw transactions | ... | ... | ... | ... |
| Clean transactions | ... | ... | ... | ... |

## Cleaning Rules

- Removed cancellation invoices where invoice_no starts with C.
- Removed rows without customer_id.
- Removed rows with quantity <= 0.
- Removed rows with unit_price <= 0.
- Removed fully duplicated transaction lines.
