"""Tests for smartstock_core. Run with: python test_core.py  (or pytest)."""
import numpy as np
import pandas as pd

import smartstock_core as core


def metrics(**kw):
    return core.compute_metrics(core.sample_inventory(), **kw).set_index("Part")


def test_fmt_inr():
    assert core.fmt_inr(0) == "₹0"
    assert core.fmt_inr(999) == "₹999"
    assert core.fmt_inr(1000) == "₹1,000"
    assert core.fmt_inr(123456) == "₹1,23,456"
    assert core.fmt_inr(12345678) == "₹1,23,45,678"
    assert core.fmt_inr(-4500) == "-₹4,500"
    assert core.fmt_inr(None) == "₹0"


def test_clean_handles_bad_data():
    raw = pd.DataFrame({
        "Part": [" Oil ", "oil", "", None, "Belt"],
        "Stock": ["5", "x", 3, 1, -4],
        "Unit Price (₹)": [10.9, np.inf, 1, 1, 2],
    })
    c = core.clean_inventory(raw)
    assert list(c["Part"]) == ["oil", "Belt"]          # case-insensitive dedupe, last wins
    assert c.loc[0, "Stock"] == 0                       # 'x' -> 0
    assert c.loc[1, "Stock"] == 0                       # negative -> 0
    assert c["Supplier"].eq("Not specified").all()
    assert list(c.columns) == core.COLUMNS
    assert core.clean_inventory(pd.DataFrame()).empty


def test_target_and_order_formula():
    m = metrics(cover_days=30)
    # Brake Pads: ceil(16*(7+30)/30)=20, max(20,13)+5=25, 25-6=19
    assert m.loc["Brake Pads", "Target Stock"] == 25
    assert m.loc["Brake Pads", "Suggested Order"] == 19
    assert m.loc["Brake Pads", "Estimated Order Cost (₹)"] == 19 * 1200
    # Stock above target -> no order
    assert m.loc["Spark Plug", "Suggested Order"] == 0
    # Floating-point trap: 6*(10+30)/30 must be exactly 8 (not 9);
    # the target is then lifted to minimum level + 1 = 9, plus safety stock 2
    assert m.loc["Clutch Plate", "Target Stock"] == 9 + 2
    assert m.loc["Oil Filter", "Target Stock"] == 24 + 4


def test_every_low_stock_part_gets_an_order():
    m = metrics()
    low = m[m["Status"].isin(core.AT_RISK)]
    assert (low["Suggested Order"] > 0).all()


def test_status_rules():
    m = metrics()
    assert m.loc["Timing Belt Kit", "Status"] == "Out of Stock"
    assert m.loc["Brake Pads", "Status"] == "Low Stock"   # 11.2 days cover > 7 days lead
    assert m.loc["Shock Absorber", "Status"] == "Critical"  # 7.5 days cover <= 8 days lead
    assert m.loc["Wiper Blade", "Status"] == "Overstock"
    assert m.loc["Engine Oil", "Status"] == "Healthy"


def test_zero_demand_has_no_cover_and_no_crash():
    df = pd.DataFrame([["A", "Other", "X", 5, 2, 0, 10, "S", 3, 1]],
                      columns=core.COLUMNS)
    m = core.compute_metrics(df)
    assert np.isnan(m.loc[0, "Days of Cover"])
    assert m.loc[0, "Status"] == "Healthy"


def test_empty_inventory():
    m = core.compute_metrics(pd.DataFrame(columns=core.COLUMNS))
    assert m.empty
    assert core.health_score(m) == 0
    assert core.purchase_orders(m).empty
    assert core.supplier_summary(m).empty
    assert core.vehicle_summary(m).empty
    assert core.generate_insights(m)[0]["level"] == "info"
    assert core.apply_filters(m, search="x").empty


def test_abc_classes():
    m = metrics()
    assert set(m["ABC"]) <= {"A", "B", "C"}
    assert (m["ABC"] == "A").any()
    ann = m["Annual Usage Value (₹)"]
    a_share = ann[m["ABC"] == "A"].sum() / ann.sum()
    assert a_share >= 0.80


def test_filters():
    m = core.compute_metrics(core.sample_inventory())
    assert set(core.apply_filters(m, search="filterpro")["Part"]) == {"Oil Filter", "Air Filter"}
    swift = core.apply_filters(m, vehicle="swift")
    assert {"Oil Filter", "Engine Oil"} <= set(swift["Part"])      # universal included
    swift_only = core.apply_filters(m, vehicle="swift", include_universal=False)
    assert "Engine Oil" not in set(swift_only["Part"])
    assert set(core.apply_filters(m, statuses=["Out of Stock"])["Part"]) == {"Timing Belt Kit"}
    assert len(core.apply_filters(m)) == len(m)
    assert core.apply_filters(m, categories=["Braking"], suppliers=["SafeStop Auto"]).shape[0] == 2


def test_purchase_orders_and_lines():
    m = core.compute_metrics(core.sample_inventory())
    po = core.purchase_orders(m)
    assert po["Order Cost (₹)"].sum() == m["Estimated Order Cost (₹)"].sum()
    sup = po.loc[0, "Supplier"]
    lines = core.po_lines(m, sup)
    assert (lines["Quantity"] > 0).all()
    assert lines["Line Total (₹)"].sum() == po.loc[0, "Order Cost (₹)"]


def test_vehicle_summary_splits_multi_vehicle_entries():
    df = pd.DataFrame([["A", "Other", "Swift, i20", 5, 2, 3, 10, "S", 3, 1]],
                      columns=core.COLUMNS)
    v = core.vehicle_summary(core.compute_metrics(df))
    assert set(v["Vehicle"]) == {"Swift", "i20"}


def test_insights_and_health():
    m = core.compute_metrics(core.sample_inventory())
    texts = " ".join(i["title"] + " " + i["text"] for i in core.generate_insights(m))
    assert "Timing Belt Kit" in texts
    assert 0 <= core.health_score(m) <= 100
    healthy = m.copy()
    healthy["Status"] = "Healthy"
    assert core.health_score(healthy) == 100


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print("PASS", t.__name__)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("FAIL", t.__name__, "->", repr(exc))
    raise SystemExit(1 if failed else 0)
