"""SmartStock AI - inventory calculations.

Pure pandas/numpy logic with no Streamlit dependency, so it can be tested
on its own. The restocking logic is rule-based (not machine learning).
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

# ---------- SCHEMA ----------
TEXT_COLS = ["Part", "Category", "Vehicle Compatibility", "Supplier"]
NUM_COLS = [
    "Stock", "Reorder Level", "Monthly Demand",
    "Unit Price (₹)", "Lead Time (Days)", "Safety Stock",
]
COLUMNS = [
    "Part", "Category", "Vehicle Compatibility", "Stock",
    "Reorder Level", "Monthly Demand", "Unit Price (₹)",
    "Supplier", "Lead Time (Days)", "Safety Stock",
]

CATEGORIES = [
    "Engine Parts", "Filters", "Lubricants", "Braking",
    "Transmission", "Ignition", "Cooling", "Electrical",
    "Suspension", "Other",
]

STATUS_ORDER = ["Out of Stock", "Critical", "Low Stock", "Healthy", "Overstock"]
AT_RISK = ["Out of Stock", "Critical", "Low Stock"]
STATUS_WEIGHT = {
    "Out of Stock": 1.0, "Critical": 0.85, "Low Stock": 0.5,
    "Overstock": 0.2, "Healthy": 0.0,
}

SAMPLE_DATA = [
    ["Engine Oil", "Lubricants", "Universal", 25, 10, 18, 450, "AutoParts Supplies", 3, 5],
    ["Oil Filter", "Filters", "Maruti Swift", 8, 15, 20, 250, "FilterPro", 5, 4],
    ["Air Filter", "Filters", "Hyundai i20", 12, 10, 12, 350, "FilterPro", 4, 3],
    ["Brake Pads", "Braking", "Honda City", 6, 12, 16, 1200, "SafeStop Auto", 7, 5],
    ["Spark Plug", "Ignition", "Maruti Swift", 30, 10, 14, 180, "PowerSpark", 3, 5],
    ["Clutch Plate", "Transmission", "Tata Nexon", 4, 8, 6, 2800, "DriveLine Parts", 10, 2],
    ["Coolant", "Cooling", "Universal", 18, 8, 10, 300, "AutoParts Supplies", 3, 4],
    ["Headlight Bulb", "Electrical", "Hyundai i20", 20, 10, 15, 220, "BrightDrive", 4, 5],
    ["Drive Belt", "Engine Parts", "Tata Nexon", 7, 10, 12, 650, "DriveLine Parts", 6, 3],
    ["Brake Fluid", "Braking", "Universal", 15, 8, 7, 400, "SafeStop Auto", 3, 3],
    ["Shock Absorber", "Suspension", "Maruti Swift", 2, 6, 8, 2200, "RideWell Components", 8, 3],
    ["Wiper Blade", "Other", "Honda City", 40, 10, 9, 280, "BrightDrive", 4, 3],
    ["Timing Belt Kit", "Engine Parts", "Hyundai i20", 0, 4, 5, 1850, "DriveLine Parts", 9, 2],
    ["Battery 12V", "Electrical", "Tata Nexon", 9, 5, 6, 5200, "PowerCell Traders", 5, 2],
]


# ---------- FORMATTING ----------
def fmt_inr(value) -> str:
    """Format a number as rupees with Indian digit grouping (₹12,34,567)."""
    try:
        n = int(round(float(value)))
    except (TypeError, ValueError):
        n = 0
    digits = str(abs(n))
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        digits = ",".join(groups + [tail])
    return f"{'-' if n < 0 else ''}₹{digits}"


# ---------- CLEANING ----------
def clean_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy copy with the standard columns and valid values."""
    df = df.copy()
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = "" if col in TEXT_COLS else 0
    df = df[COLUMNS].copy()

    for col in TEXT_COLS:
        df[col] = df[col].fillna("").astype(str).str.strip()

    for col in NUM_COLS:
        s = pd.to_numeric(df[col], errors="coerce")
        s = s.replace([np.inf, -np.inf], np.nan).fillna(0)
        df[col] = np.floor(s.clip(lower=0)).astype(int)

    df = df[df["Part"] != ""].copy()
    df["Category"] = df["Category"].replace("", "Other")
    for col in ["Vehicle Compatibility", "Supplier"]:
        df[col] = df[col].replace("", "Not specified")

    # Part names are unique, ignoring upper/lower case. The last row wins.
    df = df[~df["Part"].str.casefold().duplicated(keep="last")]
    return df.reset_index(drop=True)


def sample_inventory() -> pd.DataFrame:
    return clean_inventory(pd.DataFrame(SAMPLE_DATA, columns=COLUMNS))


# ---------- CORE CALCULATIONS ----------
def compute_metrics(df: pd.DataFrame, cover_days: int = 30,
                    overstock_days: int = 90) -> pd.DataFrame:
    """Add planning columns to an inventory table.

    Target stock = max(demand over (lead time + cover days),
                       minimum level + 1) + safety stock
    Suggested order = max(0, target stock - current stock)
    """
    d = clean_inventory(df)
    cover_days = int(cover_days)

    stock = d["Stock"]
    demand = d["Monthly Demand"]
    lead = d["Lead Time (Days)"]
    reorder = d["Reorder Level"]
    price = d["Unit Price (₹)"]

    # Integer ceiling division avoids floating-point surprises
    # (for example 0.2 * 40 = 8.000000000000002 rounding up to 9).
    lead_demand = -((-(demand * (lead + cover_days))) // 30)
    d["Target Stock"] = np.maximum(lead_demand, reorder + 1) + d["Safety Stock"]
    d["Suggested Order"] = (d["Target Stock"] - stock).clip(lower=0)
    d["Estimated Order Cost (₹)"] = d["Suggested Order"] * price
    d["Stock Value (₹)"] = stock * price
    d["Annual Usage Value (₹)"] = demand * 12 * price

    d["Days of Cover"] = (stock * 30 / demand.where(demand > 0)).round(1)

    out = (stock == 0).to_numpy()
    critical = ((stock <= reorder) & (d["Days of Cover"] <= lead)).to_numpy()
    low = (stock <= reorder).to_numpy()
    over = (d["Days of Cover"] > overstock_days).to_numpy()
    d["Status"] = np.select(
        [out, critical, low, over],
        ["Out of Stock", "Critical", "Low Stock", "Overstock"],
        default="Healthy",
    )

    # ABC classes: A = parts making up the first 80% of annual usage value,
    # B = the next 15%, C = the rest.
    annual = d["Annual Usage Value (₹)"]
    total = annual.sum()
    d["ABC"] = "C"
    if total > 0:
        ranked = annual.sort_values(ascending=False)
        share_before = (ranked.cumsum() - ranked) / total
        classes = np.where(share_before < 0.80, "A",
                           np.where(share_before < 0.95, "B", "C"))
        d["ABC"] = pd.Series(classes, index=ranked.index)

    suggested = d["Suggested Order"] > 0
    urgent = suggested & ((stock == 0) | (d["Days of Cover"] <= lead))
    d["Action"] = np.select(
        [urgent.to_numpy(), suggested.to_numpy()],
        ["Order now", "Order soon"],
        default="No action",
    )
    return d


def health_score(m: pd.DataFrame) -> int:
    """0-100 score; every unhealthy part lowers it according to severity."""
    if m.empty:
        return 0
    penalty = m["Status"].map(STATUS_WEIGHT).sum() / len(m)
    return int(round(100 * (1 - penalty)))


# ---------- FILTERING ----------
def apply_filters(m: pd.DataFrame, search: str = "", categories=None,
                  vehicle: str = "", statuses=None, suppliers=None,
                  include_universal: bool = True) -> pd.DataFrame:
    """Filter the metrics table. Empty selections mean 'no filter'."""
    if m.empty:
        return m
    mask = pd.Series(True, index=m.index)

    if search and search.strip():
        needle = search.strip().casefold()
        hit = pd.Series(False, index=m.index)
        for col in ["Part", "Category", "Vehicle Compatibility", "Supplier"]:
            hit |= m[col].str.casefold().str.contains(needle, regex=False)
        mask &= hit
    if categories:
        mask &= m["Category"].isin(categories)
    if statuses:
        mask &= m["Status"].isin(statuses)
    if suppliers:
        mask &= m["Supplier"].isin(suppliers)
    if vehicle and vehicle.strip():
        vehicles = m["Vehicle Compatibility"].str.casefold()
        hit = vehicles.str.contains(vehicle.strip().casefold(), regex=False)
        if include_universal:
            hit |= vehicles.eq("universal")
        mask &= hit
    return m[mask]


# ---------- SUMMARIES ----------
def purchase_orders(m: pd.DataFrame) -> pd.DataFrame:
    """One row per supplier for every part with a suggested order."""
    cols = ["Supplier", "Items", "Units", "Order Cost (₹)",
            "Longest Lead (Days)", "Urgent Items"]
    o = m[m["Suggested Order"] > 0]
    if o.empty:
        return pd.DataFrame(columns=cols)
    g = o.groupby("Supplier").agg(
        Items=("Part", "count"),
        Units=("Suggested Order", "sum"),
        Cost=("Estimated Order Cost (₹)", "sum"),
        Lead=("Lead Time (Days)", "max"),
        Urgent=("Action", lambda s: int((s == "Order now").sum())),
    ).reset_index()
    g.columns = cols
    return g.sort_values(["Urgent Items", "Order Cost (₹)"],
                         ascending=False).reset_index(drop=True)


def po_lines(m: pd.DataFrame, supplier: str, today: date | None = None) -> pd.DataFrame:
    """Line items of the purchase order for one supplier."""
    today = today or date.today()
    o = m[(m["Supplier"] == supplier) & (m["Suggested Order"] > 0)].copy()
    o["Expected Delivery"] = o["Lead Time (Days)"].apply(
        lambda n: (today + timedelta(days=int(n))).isoformat())
    o = o.rename(columns={
        "Suggested Order": "Quantity",
        "Estimated Order Cost (₹)": "Line Total (₹)",
    })
    return o[["Part", "Vehicle Compatibility", "Quantity", "Unit Price (₹)",
              "Line Total (₹)", "Expected Delivery"]].reset_index(drop=True)


def supplier_summary(m: pd.DataFrame) -> pd.DataFrame:
    cols = ["Supplier", "Parts", "Units", "Stock Value (₹)",
            "Avg Lead Time (Days)", "At-Risk Parts", "Restock Cost (₹)"]
    if m.empty:
        return pd.DataFrame(columns=cols)
    g = m.groupby("Supplier").agg(
        Parts=("Part", "count"),
        Units=("Stock", "sum"),
        Value=("Stock Value (₹)", "sum"),
        Lead=("Lead Time (Days)", "mean"),
        Risk=("Status", lambda s: int(s.isin(AT_RISK).sum())),
        Cost=("Estimated Order Cost (₹)", "sum"),
    ).reset_index()
    g.columns = cols
    g["Avg Lead Time (Days)"] = g["Avg Lead Time (Days)"].round(1)
    return g.sort_values("Stock Value (₹)", ascending=False).reset_index(drop=True)


def vehicle_summary(m: pd.DataFrame) -> pd.DataFrame:
    """Parts per vehicle. Entries like 'Swift, i20' count for both."""
    cols = ["Vehicle", "Parts", "Units", "Stock Value (₹)", "At-Risk Parts"]
    if m.empty:
        return pd.DataFrame(columns=cols)
    v = m.copy()
    v["Vehicle"] = v["Vehicle Compatibility"].str.split(r"[,;]").apply(
        lambda xs: [x.strip() for x in xs if x.strip()] or ["Not specified"])
    v = v.explode("Vehicle")
    v["_risk"] = v["Status"].isin(AT_RISK).astype(int)
    g = v.groupby("Vehicle").agg(
        Parts=("Part", "count"),
        Units=("Stock", "sum"),
        Value=("Stock Value (₹)", "sum"),
        Risk=("_risk", "sum"),
    ).reset_index()
    g.columns = cols
    return g.sort_values("Parts", ascending=False).reset_index(drop=True)


# ---------- INSIGHTS ----------
def generate_insights(m: pd.DataFrame, overstock_days: int = 90,
                      limit: int = 5) -> list[dict]:
    """Plain-language, rule-based observations about the inventory."""
    if m.empty:
        return [{"level": "info", "title": "No parts yet",
                 "text": "Add your first spare part to see insights."}]
    items: list[dict] = []

    urgent = m[m["Action"] == "Order now"].copy()
    urgent["_d"] = urgent["Days of Cover"].fillna(0)
    urgent = urgent.sort_values(["_d", "Estimated Order Cost (₹)"],
                                ascending=[True, False])
    for _, r in urgent.head(limit).iterrows():
        cost = fmt_inr(r["Estimated Order Cost (₹)"])
        if r["Stock"] == 0:
            text = (f"is out of stock. Reorder {r['Suggested Order']} units from "
                    f"{r['Supplier']} (delivery takes about {r['Lead Time (Days)']} "
                    f"days) for roughly {cost}.")
            level = "critical"
        else:
            text = (f"has about {r['Days of Cover']:.0f} days of stock left, but "
                    f"{r['Supplier']} needs {r['Lead Time (Days)']} days to deliver. "
                    f"Order {r['Suggested Order']} units now ({cost}).")
            level = "critical"
        items.append({"level": level, "title": r["Part"], "text": text})
    if len(urgent) > limit:
        items.append({"level": "critical", "title": f"{len(urgent) - limit} more",
                      "text": "parts also need an order right now. See the Restock Planner."})

    soon = m[m["Action"] == "Order soon"]
    if not soon.empty:
        items.append({
            "level": "warn", "title": f"{len(soon)} part(s)",
            "text": ("should be ordered soon to stay above target stock "
                     f"(about {fmt_inr(soon['Estimated Order Cost (₹)'].sum())})."),
        })

    over = m[m["Status"] == "Overstock"]
    if not over.empty:
        items.append({
            "level": "info", "title": f"{len(over)} overstocked part(s)",
            "text": (f"hold {fmt_inr(over['Stock Value (₹)'].sum())} of stock beyond "
                     f"{overstock_days} days of demand. Avoid reordering them."),
        })

    class_a = m[m["ABC"] == "A"]
    if m["Annual Usage Value (₹)"].sum() > 0 and not class_a.empty:
        items.append({
            "level": "info", "title": f"{len(class_a)} class-A part(s)",
            "text": "drive about 80% of annual usage value. Never let these run out.",
        })

    by_supplier = m.groupby("Supplier")["Stock Value (₹)"].sum()
    if len(by_supplier) > 1 and by_supplier.sum() > 0:
        top = by_supplier.idxmax()
        share = by_supplier.max() / by_supplier.sum()
        if share >= 0.40 and top != "Not specified":
            items.append({
                "level": "info", "title": top,
                "text": (f"supplies {share:.0%} of your inventory value. "
                         "Consider a backup supplier."),
            })

    if not any(i["level"] in ("critical", "warn") for i in items):
        items.insert(0, {"level": "good", "title": "All clear",
                         "text": "No parts need ordering right now."})
    return items
