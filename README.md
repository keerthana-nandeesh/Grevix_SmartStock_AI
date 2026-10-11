# 📦 SmartStock AI

**Intelligent Automobile Spare Parts Inventory Planner**

🚀 **[Launch Live Demo](https://smartstock-ai-npxafahrqez6jkgrjo2p5.streamlit.app)**

SmartStock AI is a web-based inventory planning tool designed to help automobile spare-parts shops monitor stock and make smarter purchasing decisions.

## Features
- **Dashboard**: KPI cards, inventory health score, rule-based insights, stock vs minimum level, days of cover vs supplier lead time, Pareto/ABC analysis, value treemap and donut
- **Inventory**: Add parts, edit in a table, delete rows, and view stock status and recommended actions
- **Restock Planner**: Suggested order quantities, urgency levels, budget planning, and draft purchase orders
- **Suppliers & Vehicles**: Supplier value, lead times, and vehicle-model coverage
- **Data & Export**: CSV export/import, blank template, and reset to sample data
- **Search & Filters**: Search by part, supplier, category, vehicle, or stock status

## Tech Stack
- Python
- Streamlit
- Pandas
- Plotly

## How to Run
```bash
python -m venv .venv
pip install -r requirements.txt
streamlit run app.py
```

## How the Restocking Rule Works

```text
target stock = max(demand over lead time + extra cover days, minimum level + 1) + safety stock
suggested order = max(0, target stock - current stock)
```

This is a **rule-based estimate, not a trained machine-learning model**. A future version could forecast demand using sales history and seasonality.

## Project Layout

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface |
| `smartstock_core.py` | Inventory calculations |
| `test_core.py` | Automated tests |
| `.streamlit/config.toml` | Dark theme |

**Note:** This is a prototype demonstrated using sample inventory data. Validate recommendations with real business data before operational use.
