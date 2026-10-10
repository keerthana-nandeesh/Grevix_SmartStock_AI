"""SmartStock AI - smart inventory monitoring for automobile spare-parts shops."""
from __future__ import annotations

import html
import os
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import smartstock_core as core
from smartstock_core import AT_RISK, STATUS_ORDER, fmt_inr

st.set_page_config(
    page_title="SmartStock AI", page_icon="📦", layout="wide",
    initial_sidebar_state="expanded",
)

DATA_FILE = Path(__file__).with_name("smartstock_inventory.csv")

STATUS_COLORS = {
    "Out of Stock": "#f43f5e", "Critical": "#fb923c", "Low Stock": "#facc15",
    "Healthy": "#34d399", "Overstock": "#60a5fa",
}
STATUS_ICON = {
    "Out of Stock": "🔴", "Critical": "🟠", "Low Stock": "🟡",
    "Healthy": "🟢", "Overstock": "🔵",
}
ABC_COLORS = {"A": "#f59e0b", "B": "#38bdf8", "C": "#64748b"}
PALETTE = ["#f59e0b", "#38bdf8", "#34d399", "#a78bfa", "#f43f5e",
           "#fb923c", "#22d3ee", "#e879f9", "#a3e635", "#94a3b8"]

CSS = """
<style>
.block-container{padding-top:1.4rem;padding-bottom:3rem;max-width:1400px;}
#MainMenu, footer{visibility:hidden;}
.hero{background:radial-gradient(900px 260px at 0% 0%,rgba(245,158,11,.22),transparent 60%),
radial-gradient(800px 260px at 100% 0%,rgba(56,189,248,.18),transparent 60%),
linear-gradient(135deg,#131a2e,#0f1527);
border:1px solid rgba(148,163,184,.16);border-radius:20px;padding:26px 32px;margin-bottom:18px;}
.hero-badge{display:inline-block;padding:3px 10px;border-radius:999px;font-size:.7rem;font-weight:700;
letter-spacing:.06em;text-transform:uppercase;background:rgba(245,158,11,.14);color:#fbbf24;
border:1px solid rgba(245,158,11,.35);margin-bottom:10px;}
.hero-title{font-size:2.2rem;font-weight:800;letter-spacing:-.02em;color:#f8fafc;line-height:1.15;}
.hero-sub{margin-top:6px;color:#94a3b8;font-size:1rem;}
.kpi{background:#131a2e;border:1px solid rgba(148,163,184,.16);border-radius:16px;padding:16px 18px;height:100%;}
.kpi-label{color:#94a3b8;font-size:.76rem;font-weight:600;letter-spacing:.05em;text-transform:uppercase;}
.kpi-value{color:#f8fafc;font-size:1.75rem;font-weight:800;margin-top:4px;line-height:1.1;}
.kpi-sub{color:#64748b;font-size:.8rem;margin-top:6px;}
.kpi.bad .kpi-value{color:#fb7185;}
.kpi.good .kpi-value{color:#34d399;}
.kpi.accent .kpi-value{color:#fbbf24;}
.section{margin:22px 0 4px;color:#f8fafc;font-size:1.15rem;font-weight:700;}
.section-note{color:#94a3b8;font-size:.85rem;margin-bottom:8px;}
.insight{border:1px solid rgba(148,163,184,.16);border-left-width:4px;border-radius:12px;
padding:10px 14px;margin-bottom:8px;background:#131a2e;color:#cbd5e1;font-size:.9rem;}
.insight b{color:#f8fafc;}
.insight.critical{border-left-color:#f43f5e;}
.insight.warn{border-left-color:#facc15;}
.insight.info{border-left-color:#38bdf8;}
.insight.good{border-left-color:#34d399;}
.pill{display:inline-block;padding:3px 10px;border-radius:999px;font-size:.78rem;font-weight:600;
margin:0 6px 6px 0;border:1px solid rgba(148,163,184,.25);color:#e2e8f0;background:#0f1527;}
.brand{font-size:1.25rem;font-weight:800;color:#f8fafc;margin-bottom:2px;}
.brand-sub{color:#94a3b8;font-size:.8rem;margin-bottom:10px;}
</style>
"""


# ---------- SMALL HELPERS ----------
def render(markup: str) -> None:
    """Render HTML without markdown treating indented lines as code."""
    cleaned = "\n".join(ln.strip() for ln in markup.splitlines() if ln.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def esc(value) -> str:
    return html.escape(str(value))


def section(title: str, note: str = "") -> None:
    render(f'<div class="section">{esc(title)}</div>'
           + (f'<div class="section-note">{esc(note)}</div>' if note else ""))


def kpi_card(label: str, value: str, sub: str = "", tone: str = "") -> str:
    return (f'<div class="kpi {tone}"><div class="kpi-label">{esc(label)}</div>'
            f'<div class="kpi-value">{esc(value)}</div>'
            f'<div class="kpi-sub">{esc(sub)}</div></div>')


def style_fig(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=height,
        margin=dict(l=8, r=8, t=16, b=8), font=dict(color="#cbd5e1"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)", zeroline=False)
    return fig


def show(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, width="stretch", key=key,
                    config={"displayModeBar": False})


def with_icons(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["Status"] = d["Status"].map(lambda s: f"{STATUS_ICON.get(s, '')} {s}")
    return d


# ---------- PERSISTENCE ----------
def save_inventory(df: pd.DataFrame) -> bool:
    """Write the CSV atomically so a crash cannot leave a half-written file."""
    try:
        fd, tmp = tempfile.mkstemp(dir=DATA_FILE.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            core.clean_inventory(df).to_csv(fh, index=False)
        os.replace(tmp, DATA_FILE)
        return True
    except OSError:
        st.warning("Could not save to the CSV file. Changes stay in this session only.")
        return False


def load_inventory() -> pd.DataFrame:
    if DATA_FILE.exists():
        try:
            return core.clean_inventory(pd.read_csv(DATA_FILE))
        except (ValueError, OSError, pd.errors.ParserError):
            st.error("Could not read the inventory file, so sample data is shown. "
                     "Your file was not overwritten.")
            return core.sample_inventory()
    df = core.sample_inventory()
    save_inventory(df)
    return df


def set_inventory(df: pd.DataFrame) -> None:
    """Replace the whole inventory (add, import, reset) and refresh the editor."""
    df = core.clean_inventory(df)
    st.session_state.inventory = df
    st.session_state.editor_base = df.copy()
    st.session_state.editor_version += 1
    save_inventory(df)


# ---------- STATE ----------
if "inventory" not in st.session_state:
    _inv = load_inventory()
    st.session_state.inventory = _inv
    st.session_state.editor_base = _inv.copy()
    st.session_state.editor_version = 0

if "flash" in st.session_state:
    st.toast(st.session_state.pop("flash"), icon="✅")

render(CSS)

# ---------- SIDEBAR ----------
with st.sidebar:
    render('<div class="brand">📦 SmartStock AI</div>'
           '<div class="brand-sub">Spare-parts inventory planner</div>')
    st.markdown("### 🔎 Find spare parts")
    _cur = st.session_state.inventory
    search = st.text_input("Search", placeholder="Part, supplier, category or vehicle")
    selected_categories = st.multiselect(
        "Category", sorted(_cur["Category"].unique()), placeholder="All categories")
    selected_statuses = st.multiselect(
        "Stock status", STATUS_ORDER, placeholder="All statuses")
    selected_suppliers = st.multiselect(
        "Supplier", sorted(_cur["Supplier"].unique()), placeholder="All suppliers")
    vehicle_search = st.text_input("Compatible vehicle", placeholder="e.g. Swift")
    include_universal = st.checkbox(
        "Include universal parts", value=True,
        help="When filtering by vehicle, also show parts marked 'Universal'.")

    st.markdown("### ⚙️ Planning settings")
    cover_days = st.slider(
        "Extra cover after delivery (days)", 0, 90, 30, 5,
        help="How many days of demand a delivery should cover on top of the "
             "supplier lead time. Higher values mean larger, less frequent orders.")
    overstock_days = st.slider(
        "Overstock above (days of cover)", 30, 365, 90, 15,
        help="Parts holding more days of stock than this are flagged as overstock.")

# ---------- HERO ----------
render("""
<div class="hero">
<span class="hero-badge">Prototype · sample data</span>
<div class="hero-title">SmartStock AI</div>
<div class="hero-sub">Know what to reorder, from whom, and how much it will cost, before you run out.</div>
</div>
""")

tabs = st.tabs([
    "📊 Dashboard", "🧰 Inventory", "🛒 Restock Planner",
    "🏭 Suppliers & Vehicles", "💾 Data & Export",
])

# =====================================================================
# INVENTORY TAB, part 1: add form + editable table.
# This runs first so every other tab sees the latest edits.
# =====================================================================
with tabs[1]:
    section("Add a spare part")
    with st.expander("Add a new item to inventory", expanded=False):
        with st.form("add_spare_part", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                part = st.text_input("Spare part name")
                category = st.selectbox("Category", core.CATEGORIES)
                vehicle = st.text_input(
                    "Compatible vehicle", help="Separate several models with commas.")
            with c2:
                stock = st.number_input("Current stock", min_value=0, step=1)
                reorder = st.number_input("Minimum stock level", min_value=0, value=5, step=1)
                demand = st.number_input("Estimated monthly demand", min_value=0, value=10, step=1)
            with c3:
                price = st.number_input("Unit price (₹)", min_value=0, value=100, step=1)
                supplier = st.text_input("Supplier name")
                lead_time = st.number_input("Supplier lead time (days)", min_value=0, value=5, step=1)
                safety = st.number_input("Safety stock (extra units)", min_value=0, value=3, step=1)

            add = st.form_submit_button("➕ Add spare part", type="primary", width="stretch")

        if add:
            name = part.strip()
            existing = st.session_state.inventory["Part"].str.casefold().tolist()
            if not name:
                st.error("Enter a spare part name.")
            elif name.casefold() in existing:
                st.error("This part already exists. Edit its row in the table below.")
            else:
                new_row = pd.DataFrame([{
                    "Part": name, "Category": category,
                    "Vehicle Compatibility": vehicle.strip() or "Not specified",
                    "Stock": int(stock), "Reorder Level": int(reorder),
                    "Monthly Demand": int(demand), "Unit Price (₹)": int(price),
                    "Supplier": supplier.strip() or "Not specified",
                    "Lead Time (Days)": int(lead_time), "Safety Stock": int(safety),
                }])
                set_inventory(pd.concat([st.session_state.inventory, new_row],
                                        ignore_index=True))
                st.session_state["flash"] = f"{name} added"
                st.rerun()

    section("Edit inventory",
            "Click any cell to edit. Use the table controls to add or delete rows. "
            "Every change is saved to the CSV file immediately.")

    category_options = sorted(
        set(core.CATEGORIES) | set(st.session_state.inventory["Category"]))
    edited_raw = st.data_editor(
        st.session_state.editor_base,
        num_rows="dynamic", hide_index=True, width="stretch",
        key=f"inventory_editor_{st.session_state.editor_version}",
        column_config={
            "Part": st.column_config.TextColumn("Spare Part", required=True),
            "Category": st.column_config.SelectboxColumn(
                "Category", options=category_options, required=True),
            "Vehicle Compatibility": st.column_config.TextColumn("Compatible Vehicle"),
            "Stock": st.column_config.NumberColumn("Available Units", min_value=0, step=1),
            "Reorder Level": st.column_config.NumberColumn("Minimum Stock", min_value=0, step=1),
            "Monthly Demand": st.column_config.NumberColumn("Monthly Demand", min_value=0, step=1),
            "Unit Price (₹)": st.column_config.NumberColumn("Unit Price (₹)", min_value=0, step=1, format="₹%d"),
            "Lead Time (Days)": st.column_config.NumberColumn("Lead Time (Days)", min_value=0, step=1),
            "Safety Stock": st.column_config.NumberColumn("Safety Stock", min_value=0, step=1),
        },
    )

    edited = core.clean_inventory(edited_raw)
    named_rows = int(edited_raw["Part"].fillna("").astype(str).str.strip().ne("").sum())
    if named_rows > len(edited):
        st.warning("Some rows share the same part name. Only the last one is kept.")
    if not edited.equals(st.session_state.inventory):
        st.session_state.inventory = edited
        save_inventory(edited)

# ---------- CALCULATIONS ----------
inv = st.session_state.inventory
m = core.compute_metrics(inv, cover_days, overstock_days)
v = core.apply_filters(
    m, search, selected_categories, vehicle_search,
    selected_statuses, selected_suppliers, include_universal)

with st.sidebar:
    st.divider()
    st.caption(f"Showing **{len(v)}** of **{len(m)}** parts")

# =====================================================================
# INVENTORY TAB, part 2: live stock board
# =====================================================================
with tabs[1]:
    section("Live stock board",
            "Status, days of cover and the recommended action for each part "
            "(respects the sidebar filters).")
    if v.empty:
        st.info("No parts match your filters.")
    else:
        board = with_icons(v.sort_values(
            "Status", key=lambda s: s.map({k: i for i, k in enumerate(STATUS_ORDER)})))
        st.dataframe(
            board[["Part", "Category", "Vehicle Compatibility", "Stock",
                   "Reorder Level", "Days of Cover", "Status", "ABC",
                   "Suggested Order", "Action"]],
            hide_index=True, width="stretch",
            column_config={
                "Days of Cover": st.column_config.ProgressColumn(
                    "Days of Cover", min_value=0, max_value=90, format="%.0f d"),
                "ABC": st.column_config.TextColumn(
                    "ABC", help="A = fast-moving, high-value parts. C = slow, low-value."),
            },
        )

# =====================================================================
# DASHBOARD
# =====================================================================
with tabs[0]:
    if m.empty:
        st.info("Your inventory is empty. Add a part in the Inventory tab or import a CSV.")
    else:
        at_risk = m[m["Status"].isin(AT_RISK)]
        to_order = m[m["Suggested Order"] > 0]
        units = int(m["Stock"].sum())
        value = int(m["Stock Value (₹)"].sum())
        out_n = int((m["Status"] == "Out of Stock").sum())
        crit_n = int((m["Status"] == "Critical").sum())

        cols = st.columns(5)
        cards = [
            kpi_card("Unique parts", f"{len(m)}",
                     f"{m['Category'].nunique()} categories · {m['Supplier'].nunique()} suppliers"),
            kpi_card("Units on hand", f"{units:,}", "across all parts"),
            kpi_card("Inventory value", fmt_inr(value), "at unit price", "accent"),
            kpi_card("Need attention", f"{len(at_risk)}",
                     f"{out_n} out of stock · {crit_n} critical",
                     "bad" if len(at_risk) else "good"),
            kpi_card("Restock budget", fmt_inr(to_order["Estimated Order Cost (₹)"].sum()),
                     f"{len(to_order)} parts to reorder"),
        ]
        for col, card in zip(cols, cards):
            col.markdown(card, unsafe_allow_html=True)

        score = core.health_score(m)
        gauge_color = "#34d399" if score >= 75 else "#facc15" if score >= 50 else "#f43f5e"

        left, right = st.columns([1, 1.6])
        with left:
            section("Inventory health", "Weighted by how severe each part's stock status is.")
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=score,
                number=dict(font=dict(size=46, color="#f8fafc")),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor="#64748b"),
                    bar=dict(color=gauge_color, thickness=0.3),
                    bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    steps=[
                        dict(range=[0, 50], color="rgba(244,63,94,.18)"),
                        dict(range=[50, 75], color="rgba(250,204,21,.16)"),
                        dict(range=[75, 100], color="rgba(52,211,153,.16)"),
                    ]),
            ))
            show(style_fig(gauge, 230), "chart_gauge")
            counts = m["Status"].value_counts()
            render("".join(
                f'<span class="pill">{STATUS_ICON[s]} {esc(s)} · {int(counts.get(s, 0))}</span>'
                for s in STATUS_ORDER))
        with right:
            section("Smart insights", "Rule-based observations from your current numbers.")
            for item in core.generate_insights(m, overstock_days):
                render(f'<div class="insight {esc(item["level"])}">'
                       f'<b>{esc(item["title"])}</b> {esc(item["text"])}</div>')

        if v.empty:
            st.info("No parts match your filters. Clear them in the sidebar to see charts.")
        else:
            c1, c2 = st.columns([1.7, 1])
            with c1:
                section("Stock vs minimum level",
                        "Bars are units on hand. The white tick marks the minimum stock level.")
                fig = go.Figure()
                for status in STATUS_ORDER:
                    d = v[v["Status"] == status]
                    if d.empty:
                        continue
                    fig.add_bar(
                        x=d["Part"], y=d["Stock"], name=status,
                        marker_color=STATUS_COLORS[status],
                        customdata=d[["Vehicle Compatibility", "Reorder Level"]].to_numpy(),
                        hovertemplate=("<b>%{x}</b><br>Stock: %{y}<br>Minimum: "
                                       "%{customdata[1]}<br>%{customdata[0]}<extra></extra>"))
                fig.add_scatter(
                    x=v["Part"], y=v["Reorder Level"], mode="markers", name="Minimum level",
                    marker=dict(symbol="line-ew", size=26, color="#f8fafc",
                                line=dict(width=3, color="#f8fafc")),
                    hoverinfo="skip")
                fig.update_layout(barmode="overlay")
                fig.update_xaxes(categoryorder="array", categoryarray=v["Part"].tolist(),
                                 tickangle=-35)
                show(style_fig(fig, 380), "chart_stock")
            with c2:
                section("Value by category")
                cat = v.groupby("Category", as_index=False)["Stock Value (₹)"].sum()
                cat = cat[cat["Stock Value (₹)"] > 0]
                if cat.empty:
                    st.info("No stock value to display.")
                else:
                    donut = px.pie(cat, names="Category", values="Stock Value (₹)",
                                   hole=0.58, color_discrete_sequence=PALETTE)
                    donut.update_traces(textinfo="percent", textposition="inside",
                                        marker=dict(line=dict(color="#0b1020", width=2)))
                    donut.add_annotation(
                        text=f"<b>{esc(fmt_inr(cat['Stock Value (₹)'].sum()))}</b>",
                        showarrow=False, font=dict(size=18, color="#f8fafc"))
                    donut.update_layout(showlegend=True, legend=dict(orientation="v"))
                    show(style_fig(donut, 380), "chart_category")

            c3, c4 = st.columns(2)
            with c3:
                section("Days of cover vs supplier lead time",
                        "Bars: days the stock will last. Diamonds: days the supplier needs to deliver. "
                        "A bar shorter than its diamond means a stock-out before the next delivery.")
                cover = v[v["Days of Cover"].notna()].sort_values("Days of Cover").head(12)
                if cover.empty:
                    st.info("No demand data to calculate days of cover.")
                else:
                    fig = go.Figure()
                    fig.add_bar(
                        y=cover["Part"], x=cover["Days of Cover"], orientation="h",
                        name="Days of cover",
                        marker_color=[STATUS_COLORS[s] for s in cover["Status"]],
                        hovertemplate="<b>%{y}</b><br>%{x} days of cover<extra></extra>")
                    fig.add_scatter(
                        y=cover["Part"], x=cover["Lead Time (Days)"], mode="markers",
                        name="Supplier lead time",
                        marker=dict(symbol="diamond", size=12, color="#f8fafc",
                                    line=dict(width=1, color="#0b1020")),
                        hovertemplate="<b>%{y}</b><br>%{x} days lead time<extra></extra>")
                    fig.update_yaxes(categoryorder="array",
                                     categoryarray=cover["Part"].tolist(), autorange="reversed")
                    fig.update_xaxes(title="Days")
                    show(style_fig(fig, 400), "chart_cover")
            with c4:
                section("Pareto: which parts matter most",
                        "Annual usage value per part. Colour shows its ABC class "
                        "(A = about 80% of value). The line is the cumulative share.")
                par = v[v["Annual Usage Value (₹)"] > 0].sort_values(
                    "Annual Usage Value (₹)", ascending=False)
                if par.empty:
                    st.info("No demand data to build the Pareto chart.")
                else:
                    cum = par["Annual Usage Value (₹)"].cumsum() / par["Annual Usage Value (₹)"].sum() * 100
                    fig = go.Figure()
                    fig.add_bar(
                        x=par["Part"], y=par["Annual Usage Value (₹)"], name="Annual usage value",
                        marker_color=[ABC_COLORS[c] for c in par["ABC"]],
                        customdata=par["ABC"].to_numpy(),
                        hovertemplate="<b>%{x}</b><br>Class %{customdata}<br>₹%{y:,.0f} per year<extra></extra>")
                    fig.add_scatter(
                        x=par["Part"], y=cum, yaxis="y2", mode="lines+markers",
                        name="Cumulative %", line=dict(color="#f8fafc", width=2),
                        hovertemplate="%{y:.0f}% cumulative<extra></extra>")
                    fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 105],
                                                  ticksuffix="%", showgrid=False))
                    fig.update_xaxes(tickangle=-35)
                    show(style_fig(fig, 400), "chart_pareto")

            c5, c6 = st.columns(2)
            with c5:
                section("Where the money sits", "Inventory value by category and part.")
                tree = v[v["Stock Value (₹)"] > 0]
                if tree.empty:
                    st.info("No stock value to display.")
                else:
                    fig = px.treemap(tree, path=["Category", "Part"], values="Stock Value (₹)",
                                     color_discrete_sequence=PALETTE)
                    fig.update_traces(root_color="rgba(0,0,0,0)",
                                      marker=dict(line=dict(color="#0b1020", width=2)))
                    show(style_fig(fig, 360), "chart_treemap")
            with c6:
                section("Restock spend by supplier")
                po = core.purchase_orders(v)
                if po.empty:
                    st.success("No restocking suggested for the current filters.")
                else:
                    fig = go.Figure(go.Bar(
                        x=po["Supplier"], y=po["Order Cost (₹)"],
                        marker_color=["#f43f5e" if u > 0 else "#f59e0b" for u in po["Urgent Items"]],
                        customdata=po[["Items", "Urgent Items"]].to_numpy(),
                        hovertemplate=("<b>%{x}</b><br>₹%{y:,.0f}<br>%{customdata[0]} items · "
                                       "%{customdata[1]} urgent<extra></extra>")))
                    fig.update_xaxes(tickangle=-20)
                    show(style_fig(fig, 360), "chart_supplier_spend")
                    st.caption("Red bars include at least one urgent item.")

            section("Low-stock alerts")
            alerts = v[v["Status"].isin(AT_RISK)].sort_values(
                "Status", key=lambda s: s.map({k: i for i, k in enumerate(STATUS_ORDER)}))
            if alerts.empty:
                st.success("No low-stock items in the current view.")
            else:
                st.warning(f"{len(alerts)} item(s) need attention.")
                st.dataframe(
                    with_icons(alerts)[["Part", "Vehicle Compatibility", "Stock", "Reorder Level",
                                        "Days of Cover", "Status", "Supplier", "Suggested Order"]],
                    hide_index=True, width="stretch",
                    column_config={"Days of Cover": st.column_config.NumberColumn(format="%.1f")})

# =====================================================================
# RESTOCK PLANNER
# =====================================================================
with tabs[2]:
    section("Restock planner",
            "Suggested quantities to bring each part up to its target stock.")
    f1, f2 = st.columns([1, 1])
    with f1:
        only_urgent = st.toggle("Only urgent orders", value=False,
                                help="Show only parts that will run out before a delivery can arrive.")
    with f2:
        budget = st.number_input("Available budget (₹, 0 = no limit)", min_value=0,
                                 value=0, step=1000)

    orders = v[v["Suggested Order"] > 0].copy()
    if only_urgent:
        orders = orders[orders["Action"] == "Order now"]
    orders["_rank"] = orders["Action"].map({"Order now": 0, "Order soon": 1})
    orders = orders.sort_values(["_rank", "Days of Cover"], na_position="last")

    if orders.empty:
        st.success("No restocking is suggested for the current selection.")
    else:
        total_cost = int(orders["Estimated Order Cost (₹)"].sum())
        cols = st.columns(3)
        cols[0].markdown(kpi_card("Parts to order", f"{len(orders)}",
                                  f"{int((orders['Action'] == 'Order now').sum())} urgent",
                                  "bad" if (orders["Action"] == "Order now").any() else ""),
                         unsafe_allow_html=True)
        cols[1].markdown(kpi_card("Units to order", f"{int(orders['Suggested Order'].sum()):,}",
                                  f"from {orders['Supplier'].nunique()} suppliers"),
                         unsafe_allow_html=True)
        cols[2].markdown(kpi_card("Estimated cost", fmt_inr(total_cost),
                                  "at current unit prices", "accent"),
                         unsafe_allow_html=True)

        table_cols = ["Part", "Supplier", "Stock", "Target Stock", "Suggested Order",
                      "Unit Price (₹)", "Estimated Order Cost (₹)", "Lead Time (Days)",
                      "Days of Cover", "Action"]
        if budget > 0:
            cum = orders["Estimated Order Cost (₹)"].cumsum()
            orders["Within Budget"] = np.where(cum <= budget, "✅ Yes", "⏳ Later")
            table_cols.append("Within Budget")
            affordable = orders[orders["Within Budget"] == "✅ Yes"]
            st.info(f"Your budget of {fmt_inr(budget)} covers {len(affordable)} of "
                    f"{len(orders)} parts, most urgent first "
                    f"({fmt_inr(affordable['Estimated Order Cost (₹)'].sum())}).")

        st.dataframe(
            orders[table_cols], hide_index=True, width="stretch",
            column_config={
                "Unit Price (₹)": st.column_config.NumberColumn(format="₹%d"),
                "Estimated Order Cost (₹)": st.column_config.NumberColumn(
                    "Order Cost (₹)", format="₹%d"),
                "Days of Cover": st.column_config.NumberColumn(format="%.1f"),
            })

        with st.expander("How is this calculated?"):
            st.markdown(
                f"""
**Target stock** = the larger of
(a) demand over *lead time + {cover_days} extra days*, and
(b) minimum stock level + 1 — then add **safety stock**.

**Suggested order** = target stock − current stock (never below zero).

**Order now** means stock is zero, or will run out before a delivery could arrive.
**Order soon** means a top-up is needed but there is still time.

Demand is spread evenly over 30 days. This is a rule-based estimate, not a trained
forecast. Confirm real demand and delivery times with your suppliers.
""")

        section("Purchase orders by supplier",
                "One draft order per supplier, ready to download and send.")
        pos = core.purchase_orders(orders)
        for i, row in pos.iterrows():
            label = (f"{row['Supplier']} · {row['Items']} item(s) · "
                     f"{fmt_inr(row['Order Cost (₹)'])}"
                     + (f" · {row['Urgent Items']} urgent" if row["Urgent Items"] else ""))
            with st.expander(label):
                lines = core.po_lines(orders, row["Supplier"])
                st.dataframe(lines, hide_index=True, width="stretch",
                             column_config={
                                 "Unit Price (₹)": st.column_config.NumberColumn(format="₹%d"),
                                 "Line Total (₹)": st.column_config.NumberColumn(format="₹%d"),
                             })
                safe = "".join(ch if ch.isalnum() else "_" for ch in row["Supplier"]).strip("_")
                st.download_button(
                    f"Download order for {row['Supplier']}",
                    data=lines.to_csv(index=False).encode("utf-8"),
                    file_name=f"order_{safe}_{date.today().isoformat()}.csv",
                    mime="text/csv", key=f"po_download_{i}")

        st.download_button(
            "⬇️ Download full restock plan (CSV)",
            data=orders[table_cols].to_csv(index=False).encode("utf-8"),
            file_name=f"restock_plan_{date.today().isoformat()}.csv",
            mime="text/csv", type="primary", key="download_plan")

# =====================================================================
# SUPPLIERS & VEHICLES
# =====================================================================
with tabs[3]:
    if m.empty:
        st.info("Add parts to see supplier and vehicle summaries.")
    else:
        s1, s2 = st.columns(2)
        sup = core.supplier_summary(m)
        with s1:
            section("Suppliers",
                    "Inventory value per supplier, coloured by average delivery time.")
            fig = px.bar(sup, x="Stock Value (₹)", y="Supplier", orientation="h",
                         color="Avg Lead Time (Days)",
                         color_continuous_scale=["#34d399", "#facc15", "#f43f5e"])
            fig.update_yaxes(categoryorder="total ascending")
            fig.update_layout(coloraxis_colorbar=dict(title="Lead days"))
            show(style_fig(fig, 340), "chart_suppliers")
        with s2:
            section("Vehicle coverage",
                    "Stock value per vehicle model. Parts that fit several models count for each.")
            veh = core.vehicle_summary(m)
            fig = px.bar(veh, x="Vehicle", y="Stock Value (₹)",
                         color="At-Risk Parts",
                         color_continuous_scale=["#34d399", "#facc15", "#f43f5e"])
            fig.update_xaxes(tickangle=-25)
            show(style_fig(fig, 340), "chart_vehicles")

        section("Supplier details")
        st.dataframe(sup, hide_index=True, width="stretch", column_config={
            "Stock Value (₹)": st.column_config.NumberColumn(format="₹%d"),
            "Restock Cost (₹)": st.column_config.NumberColumn(format="₹%d"),
        })
        section("Vehicle details")
        st.dataframe(veh, hide_index=True, width="stretch", column_config={
            "Stock Value (₹)": st.column_config.NumberColumn(format="₹%d"),
        })

# =====================================================================
# DATA & EXPORT
# =====================================================================
with tabs[4]:
    section("Export")
    e1, e2, e3 = st.columns(3)
    e1.download_button(
        "⬇️ Inventory (CSV)", data=inv.to_csv(index=False).encode("utf-8"),
        file_name="smartstock_inventory.csv", mime="text/csv",
        width="stretch", key="dl_inventory",
        help="Your editable inventory. Can be imported again later.")
    e2.download_button(
        "⬇️ Full analysis (CSV)", data=m.to_csv(index=False).encode("utf-8"),
        file_name="smartstock_analysis.csv", mime="text/csv",
        width="stretch", key="dl_analysis",
        help="Inventory plus status, days of cover, ABC class and suggested orders.")
    e3.download_button(
        "⬇️ Blank template (CSV)",
        data=pd.DataFrame(columns=core.COLUMNS).to_csv(index=False).encode("utf-8"),
        file_name="smartstock_template.csv", mime="text/csv",
        width="stretch", key="dl_template")

    section("Import",
            "Upload a CSV with the same columns as the template. Only 'Part' is required.")
    upload = st.file_uploader("Choose a CSV file", type=["csv"])
    mode = st.radio("How should it be applied?",
                    ["Merge: update matching parts and add new ones",
                     "Replace: use only the uploaded parts"])
    if upload is not None:
        try:
            try:
                incoming = pd.read_csv(upload, encoding="utf-8-sig")
            except UnicodeDecodeError:
                upload.seek(0)
                incoming = pd.read_csv(upload, encoding="latin-1")
        except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            st.error(f"Could not read that file: {exc}")
        else:
            if "Part" not in incoming.columns:
                st.error("The file needs a 'Part' column.")
            else:
                missing = [c for c in core.COLUMNS if c not in incoming.columns]
                if missing:
                    st.warning("Missing columns will use defaults: " + ", ".join(missing))
                cleaned = core.clean_inventory(incoming)
                st.success(f"Found {len(cleaned)} valid part(s).")
                st.dataframe(cleaned.head(10), hide_index=True, width="stretch")
                if st.button("Apply import", type="primary"):
                    if mode.startswith("Merge"):
                        combined = pd.concat([inv, cleaned], ignore_index=True)
                    else:
                        combined = cleaned
                    set_inventory(combined)
                    st.session_state["flash"] = f"Imported {len(cleaned)} part(s)"
                    st.rerun()

    section("Reset")
    confirm = st.checkbox("I understand this replaces all current data with the sample data.")
    if st.button("Reset to sample data", disabled=not confirm):
        set_inventory(core.sample_inventory())
        st.session_state["flash"] = "Sample data restored"
        st.rerun()

st.caption(
    "SmartStock AI · Prototype with illustrative sample data. Recommendations are rule-based, "
    "not machine-learning forecasts. Verify vehicle compatibility, supplier details and demand "
    "before ordering.")
