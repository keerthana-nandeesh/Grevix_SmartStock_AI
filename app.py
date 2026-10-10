
import os
import math
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="SmartStock AI", page_icon="📦", layout="wide")

# ---------- CONFIGURATION ----------
DATA_FILE = "smartstock_inventory.csv"

COLUMNS = [
    "Part", "Category", "Vehicle Compatibility", "Stock",
    "Reorder Level", "Monthly Demand", "Unit Price (₹)",
    "Supplier", "Lead Time (Days)", "Safety Stock"
]

CATEGORIES = [
    "Engine Parts", "Filters", "Lubricants", "Braking",
    "Transmission", "Ignition", "Cooling", "Electrical",
    "Suspension", "Other"
]

SAMPLE_DATA = [
    ["Engine Oil", "Lubricants", "Universal", 25, 10, 18, 450,
     "AutoParts Supplies", 3, 5],
    ["Oil Filter", "Filters", "Maruti Swift", 8, 15, 20, 250,
     "FilterPro", 5, 4],
    ["Air Filter", "Filters", "Hyundai i20", 12, 10, 12, 350,
     "FilterPro", 4, 3],
    ["Brake Pads", "Braking", "Honda City", 6, 12, 16, 1200,
     "SafeStop Auto", 7, 5],
    ["Spark Plug", "Ignition", "Maruti Swift", 30, 10, 14, 180,
     "PowerSpark", 3, 5],
    ["Clutch Plate", "Transmission", "Tata Nexon", 4, 8, 6, 2800,
     "DriveLine Parts", 10, 2],
    ["Coolant", "Cooling", "Universal", 18, 8, 10, 300,
     "AutoParts Supplies", 3, 4],
    ["Headlight Bulb", "Electrical", "Hyundai i20", 20, 10, 15, 220,
     "BrightDrive", 4, 5],
    ["Drive Belt", "Engine Parts", "Tata Nexon", 7, 10, 12, 650,
     "DriveLine Parts", 6, 3],
    ["Brake Fluid", "Braking", "Universal", 15, 8, 7, 400,
     "SafeStop Auto", 3, 3],
]


def clean_inventory(df):
    df = df.copy()
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = "" if col in [
                "Part", "Category", "Vehicle Compatibility", "Supplier"
            ] else 0

    df = df[COLUMNS]
    text_cols = [
        "Part", "Category", "Vehicle Compatibility", "Supplier"
    ]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str).str.strip()

    numeric_cols = [
        "Stock", "Reorder Level", "Monthly Demand",
        "Unit Price (₹)", "Lead Time (Days)", "Safety Stock"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df[col] = df[col].clip(lower=0).apply(math.floor).astype(int)

    df = df[df["Part"] != ""]
    df = df.drop_duplicates(subset=["Part"], keep="last")
    return df.reset_index(drop=True)


def save_inventory(df):
    clean_inventory(df).to_csv(DATA_FILE, index=False)


def load_inventory():
    if os.path.exists(DATA_FILE):
        try:
            return clean_inventory(pd.read_csv(DATA_FILE))
        except (ValueError, OSError, pd.errors.ParserError):
            st.error("Could not read the inventory file. Check the CSV file.")
            return clean_inventory(pd.DataFrame(SAMPLE_DATA, columns=COLUMNS))
    df = clean_inventory(pd.DataFrame(SAMPLE_DATA, columns=COLUMNS))
    save_inventory(df)
    return df


# ---------- LOAD INVENTORY ----------
if "inventory" not in st.session_state:
    st.session_state.inventory = load_inventory()

# ---------- HEADER ----------
st.title("📦 SmartStock AI")
st.write("Smart inventory monitoring and stock planning")
st.caption("Automobile spare-parts inventory management")

# ---------- ADD SPARE PART ----------
st.subheader("➕ Add Spare Part")

with st.expander("Add a new item to inventory", expanded=True):
    with st.form("add_spare_part", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)

        with c1:
            part = st.text_input("Spare part name")
            category = st.selectbox("Category", CATEGORIES)
            vehicle = st.text_input("Compatible vehicle")

        with c2:
            stock = st.number_input("Current stock", min_value=0, step=1)
            reorder = st.number_input(
                "Minimum stock level", min_value=0, value=5, step=1
            )
            demand = st.number_input(
                "Estimated monthly demand", min_value=0, value=10, step=1
            )

        with c3:
            price = st.number_input("Unit price (₹)", min_value=0, value=100)
            supplier = st.text_input("Supplier name")
            lead_time = st.number_input(
                "Supplier lead time (days)", min_value=0, value=5, step=1
            )
            safety = st.number_input(
                "Safety stock (extra units)", min_value=0, value=3, step=1
            )

        add = st.form_submit_button("➕ Add Spare Part", use_container_width=True)

        if add:
            if not part.strip():
                st.error("Enter a spare part name.")
            elif part.strip().casefold() in (
                st.session_state.inventory["Part"].str.casefold().tolist()
            ):
                st.error("This part already exists. Update its existing row.")
            else:
                new_row = pd.DataFrame([{
                    "Part": part.strip(),
                    "Category": category,
                    "Vehicle Compatibility": vehicle.strip() or "Not specified",
                    "Stock": int(stock),
                    "Reorder Level": int(reorder),
                    "Monthly Demand": int(demand),
                    "Unit Price (₹)": int(price),
                    "Supplier": supplier.strip() or "Not specified",
                    "Lead Time (Days)": int(lead_time),
                    "Safety Stock": int(safety)
                }])

                st.session_state.inventory = pd.concat(
                    [st.session_state.inventory, new_row],
                    ignore_index=True
                )
                save_inventory(st.session_state.inventory)
                st.success(f"{part.strip()} added successfully!")
                st.rerun()

# ---------- SEARCH AND FILTERS ----------
st.sidebar.header("🔎 Find Spare Parts")

search = st.sidebar.text_input("Search part or supplier")

available_categories = sorted(
    st.session_state.inventory["Category"].dropna().unique().tolist()
)
selected_categories = st.sidebar.multiselect(
    "Category",
    available_categories,
    default=available_categories
)
vehicle_search = st.sidebar.text_input("Compatible vehicle")

# ---------- EDIT INVENTORY ----------
st.divider()
st.subheader("🛠️ Inventory Management")
st.caption(
    "Edit stock and other details directly. Use the table controls "
    "to add or delete rows. Changes are saved to the local CSV file."
)

edited = st.data_editor(
    st.session_state.inventory,
    num_rows="dynamic",
    hide_index=True,
    width="stretch",
    key="inventory_editor",
    column_config={
        "Part": st.column_config.TextColumn("Spare Part", required=True),
        "Category": st.column_config.SelectboxColumn(
            "Category", options=CATEGORIES, required=True
        ),
        "Stock": st.column_config.NumberColumn(
            "Available Units", min_value=0, step=1
        ),
        "Reorder Level": st.column_config.NumberColumn(
            "Minimum Stock", min_value=0, step=1
        ),
        "Monthly Demand": st.column_config.NumberColumn(
            "Monthly Demand", min_value=0, step=1
        ),
        "Unit Price (₹)": st.column_config.NumberColumn(
            "Unit Price (₹)", min_value=0, step=1
        ),
        "Lead Time (Days)": st.column_config.NumberColumn(
            "Lead Time (Days)", min_value=0, step=1
        ),
        "Safety Stock": st.column_config.NumberColumn(
            "Safety Stock", min_value=0, step=1
        )
    }
)

edited = clean_inventory(edited)

if not edited.equals(clean_inventory(st.session_state.inventory)):
    st.session_state.inventory = edited
    save_inventory(edited)
else:
    st.session_state.inventory = edited

# ---------- CALCULATIONS ----------
df = st.session_state.inventory.copy()

# Rule-based restocking: cover lead-time demand plus safety stock.
# This is an estimate, not a trained machine-learning forecast.
df["Status"] = df.apply(
    lambda r: "Low Stock" if r["Stock"] <= r["Reorder Level"]
    else "In Stock",
    axis=1
)

df["Stock Value (₹)"] = df["Stock"] * df["Unit Price (₹)"]

df["Suggested Order"] = df.apply(
    lambda r: max(
        0,
        math.ceil(r["Monthly Demand"] / 30 * r["Lead Time (Days)"])
        + r["Safety Stock"] - r["Stock"]
    ),
    axis=1
)

df["Estimated Order Cost (₹)"] = (
    df["Suggested Order"] * df["Unit Price (₹)"]
)

# ---------- FILTERED VIEW ----------
view = df.copy()

if search:
    view = view[
        view.apply(
            lambda row: search.casefold() in " ".join(
                row.astype(str)
            ).casefold(),
            axis=1
        )
    ]

view = view[view["Category"].isin(selected_categories)]

if vehicle_search:
    view = view[
        view["Vehicle Compatibility"].str.contains(
            vehicle_search, case=False, na=False
        )
    ]

# ---------- METRICS ----------
st.divider()
st.subheader("📊 Inventory Overview")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Unique Parts", len(df))
m2.metric("Total Units", f"{int(df['Stock'].sum()):,}")
m3.metric("Low-Stock Parts", int((df["Status"] == "Low Stock").sum()))
m4.metric("Inventory Value", f"₹{df['Stock Value (₹)'].sum():,.0f}")

# ---------- CHARTS ----------
left, right = st.columns(2)

with left:
    st.subheader("Stock by Spare Part")
    if not view.empty:
        fig = px.bar(
            view,
            x="Part",
            y="Stock",
            color="Status",
            hover_data=["Category", "Vehicle Compatibility"],
            color_discrete_map={
                "In Stock": "#16a34a",
                "Low Stock": "#dc2626"
            }
        )
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No parts match your filters.")

with right:
    st.subheader("Inventory Value by Category")
    if not view.empty:
        category_value = view.groupby(
            "Category", as_index=False
        )["Stock Value (₹)"].sum()

        fig2 = px.pie(
            category_value,
            names="Category",
            values="Stock Value (₹)",
            hole=0.45
        )
        st.plotly_chart(fig2, width="stretch")
    else:
        st.info("No category data to display.")

# ---------- LOW STOCK ALERTS ----------
st.divider()
st.subheader("🚨 Low-Stock Alerts")

alerts = view[view["Status"] == "Low Stock"]

if alerts.empty:
    st.success("No low-stock items in the current filtered view.")
else:
    st.warning(f"{len(alerts)} item(s) need attention.")
    st.dataframe(
        alerts[
            ["Part", "Vehicle Compatibility", "Stock",
             "Reorder Level", "Supplier"]
        ],
        hide_index=True,
        width="stretch"
    )

# ---------- RESTOCKING ----------
st.subheader("📦 Restocking Recommendations")
st.caption(
    "Suggested quantity = lead-time demand + safety stock − current stock, "
    "with a minimum of zero. Confirm actual demand and supplier delivery times."
)

orders = view[view["Suggested Order"] > 0].copy()

if orders.empty:
    st.success("No restocking is suggested for the current filtered items.")
else:
    st.dataframe(
        orders[
            ["Part", "Stock", "Monthly Demand", "Lead Time (Days)",
             "Safety Stock", "Suggested Order", "Supplier",
             "Unit Price (₹)", "Estimated Order Cost (₹)"]
        ],
        hide_index=True,
        width="stretch"
    )
    st.metric(
        "Estimated Restocking Cost",
        f"₹{orders['Estimated Order Cost (₹)'].sum():,.0f}"
    )

# ---------- EXPORT ----------
st.divider()
st.subheader("📥 Export Inventory")

st.download_button(
    "Download Inventory as CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="smartstock_inventory.csv",
    mime="text/csv"
)

st.caption(
    "SmartStock AI | Prototype with illustrative sample data. "
    "Verify vehicle compatibility, supplier details and demand before ordering."
)
