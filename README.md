# 📦 SmartStock AI

Smart inventory monitoring and stock planning for automobile spare-parts shops.

## Features
- **Dashboard**: KPI cards, inventory health score, rule-based insights, stock vs minimum level, days of cover vs supplier lead time, Pareto/ABC analysis, value treemap and donut
- **Inventory**: add parts, edit in a table, delete rows, live stock board with status and recommended action
- **Restock Planner**: suggested order quantities, urgent vs soon, budget what-if, one draft purchase order per supplier (CSV)
- **Suppliers & Vehicles**: value and lead time per supplier, coverage per vehicle model
- **Data & Export**: CSV export, CSV import (merge or replace), blank template, reset to sample data
- **Search & filters**: part, supplier, category, vehicle (with universal parts), stock status
- **Persistence**: every change is saved to `smartstock_inventory.csv` next to `app.py`

## Run it
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## How the restocking rule works
```
target stock    = max( demand over (lead time + extra cover days), minimum level + 1 ) + safety stock
suggested order = max(0, target stock - current stock)
```
- **Order now**: stock is zero, or will run out before a delivery can arrive
- **Order soon**: a top-up is needed but there is still time
- Status: Out of Stock, Critical (at/below minimum and runs out before delivery), Low Stock (at/below minimum), Healthy, Overstock (more than N days of cover)
- ABC class: A = parts making up the first 80% of annual usage value, B = next 15%, C = rest

This is a **rule-based estimate, not a trained machine-learning model**. A future version could forecast demand from sales history and seasonality.

## Project layout
| File | Purpose |
|---|---|
| `app.py` | Streamlit interface |
| `smartstock_core.py` | Calculations (no Streamlit dependency) |
| `test_core.py` | Tests: `python test_core.py` or `pytest` |
| `.streamlit/config.toml` | Dark theme |

## Important
Sample inventory, prices, suppliers and demand figures are illustrative. Replace them with verified business data before real-world use.
