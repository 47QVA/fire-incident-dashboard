'''
Fire Incident Analytics Dashboard (Streamlit)
Run:  streamlit run fire_dashboard_app.py
Needs: incidents_clean.csv  (+ optional fire_ml_models/ outputs from the ML notebook)
'''
import io
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Fire Incident Dashboard", page_icon="🔥", layout="wide")

DATA_DIR = Path(__file__).parent
ML_DIR = DATA_DIR / "fire_ml_models"
RESPONSE_TARGET_MIN = 3          # target minutes from call to fire-crew response
NATURE_COLORS = {"EQUIPMENT": "#C0392B", "BUILDING": "#E67E22", "3RD PARTY": "#2E86C1", "BUSH/SCRAP": "#27AE60"}
SEV_COLORS = {"High": "#C0392B", "Medium": "#F39C12", "Low": "#27AE60"}
MONTH_ORDER = ["January", "February", "March", "April", "May", "June", "July", "August",
               "September", "October", "November", "December"]


@st.cache_data
def load():
    inc = pd.read_csv(DATA_DIR / "incidents_clean.csv", parse_dates=["incident_date", "call_datetime"])
    inc.columns = inc.columns.str.replace("\ufeff", "")
    inc["month_name"] = pd.Categorical(inc["month_name"], MONTH_ORDER, ordered=True)
    inc["ym"] = inc["incident_date"].dt.to_period("M").astype(str)
    ml = {}
    for k, f in {"area_risk": "area_risk_30d.csv", "metrics": "model_metrics.csv",
                 "inc_pred": "incident_predictions.csv"}.items():
        p = ML_DIR / f
        if p.exists():
            d = pd.read_csv(p); d.columns = d.columns.str.replace("\ufeff", ""); ml[k] = d
    return inc, ml


def equipment_risk(df, ref_date):
    e = df.dropna(subset=["primary_tag_base"]).copy()
    if e.empty:
        return pd.DataFrame()
    sev = {"Low": 1, "Medium": 2, "High": 3}
    e["sev_pts"] = e["severity_level"].map(sev).fillna(1)
    g = e.groupby("primary_tag_base").agg(
        area=("area", "first"), equipment_type=("equipment_type", "first"),
        fires=("incident_id", "count"), avg_severity=("sev_pts", "mean"),
        injuries=("injury_reported", "sum"), last_fire=("incident_date", "max"),
        first_fire=("incident_date", "min")).reset_index()
    g["days_since_last"] = (ref_date - g["last_fire"]).dt.days
    g["recency_factor"] = 0.5 ** (g["days_since_last"] / 90)
    g["risk_score"] = (g["fires"] * 10 + g["avg_severity"] * 10 + g["injuries"] * 15 + g["recency_factor"] * 20).round(1)
    g["risk_rank"] = g["risk_score"].rank(ascending=False, method="min").astype(int)
    g["risk_band"] = pd.cut(g["risk_score"], [-1, 30, 50, 1e9], labels=["Low", "Medium", "High"])
    return g.sort_values("risk_rank")


def to_excel(sheets):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for n, d in sheets.items():
            d.to_excel(xw, sheet_name=n[:31], index=False)
    return buf.getvalue()


inc, ml = load()

# ---------------- Sidebar filters ----------------
st.sidebar.title("🔥 Filters")
dmin, dmax = inc["incident_date"].min().date(), inc["incident_date"].max().date()
dr = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
zones = st.sidebar.multiselect("Zone", sorted(inc["zone"].unique()), default=sorted(inc["zone"].unique()))
natures = st.sidebar.multiselect("Fire nature", sorted(inc["fire_nature"].unique()), default=sorted(inc["fire_nature"].unique()))
sevs = st.sidebar.multiselect("Severity", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
start, end = (dr if isinstance(dr, (list, tuple)) and len(dr) == 2 else (dmin, dmax))
f = inc[(inc["incident_date"].dt.date >= start) & (inc["incident_date"].dt.date <= end)
        & inc["zone"].isin(zones) & inc["fire_nature"].isin(natures) & inc["severity_level"].isin(sevs)]
if f.empty:
    st.warning("No incidents for the selected filters."); st.stop()

st.title("Fire Emergency Analytics Dashboard")
st.caption(f"{len(f)} incidents | {start:%d %b %Y} – {end:%d %b %Y} | response target ≤ {RESPONSE_TARGET_MIN} min")

tabs = st.tabs(["📊 Executive KPIs", "📈 Trends", "🗺️ Heat maps", "⏱️ Response scorecard",
                "🔍 Cause analysis", "⚙️ Equipment risk", "🤖 ML risk forecast", "📥 Reports"])

# ---------------- Executive KPIs ----------------
with tabs[0]:
    last_m = f["ym"].max()
    months = sorted(f["ym"].unique())
    prev_m = months[-2] if len(months) > 1 else None
    resp = f["response_minutes"].dropna()
    c = st.columns(4)
    c[0].metric("Total fires", len(f),
                f"{(f.ym == last_m).sum() - (f.ym == prev_m).sum():+d} vs prev month" if prev_m else None, delta_color="inverse")
    c[1].metric("Avg response (min)", f"{resp.mean():.1f}", f"{(resp <= RESPONSE_TARGET_MIN).mean():.0%} within target")
    c[2].metric("Out before fire crew", f"{f['out_before_fire_crew'].mean():.0%}")
    c[3].metric("Injury incidents", int(f["injury_reported"].sum()))
    c = st.columns(4)
    c[0].metric("High-severity fires", int((f["severity_level"] == "High").sum()))
    c[1].metric("Repeat-equipment fires", int(f["is_repeat_equipment"].sum()))
    c[2].metric("Top area", f["area"].value_counts().idxmax(), f"{f['area'].value_counts().max()} fires", delta_color="off")
    c[3].metric("Top cause", f["cause_category"].value_counts().idxmax(), delta_color="off")
    a, b = st.columns(2)
    a.plotly_chart(px.pie(f, names="fire_nature", hole=0.55, color="fire_nature", color_discrete_map=NATURE_COLORS,
                          title="Fires by nature"), use_container_width=True)
    b.plotly_chart(px.pie(f, names="severity_level", hole=0.55, color="severity_level", color_discrete_map=SEV_COLORS,
                          title="Fires by severity"), use_container_width=True)

# ---------------- Trends ----------------
with tabs[1]:
    m = f.groupby(["ym", "fire_nature"]).size().reset_index(name="fires")
    st.plotly_chart(px.bar(m, x="ym", y="fires", color="fire_nature", color_discrete_map=NATURE_COLORS,
                           title="Monthly fires by nature", labels={"ym": "Month"}), use_container_width=True)
    t = f.groupby("ym").size().reset_index(name="fires")
    t["cumulative"] = t["fires"].cumsum(); t["3-mo avg"] = t["fires"].rolling(3, min_periods=1).mean()
    fig = go.Figure()
    fig.add_bar(x=t["ym"], y=t["fires"], name="Fires", marker_color="#E6B0AA")
    fig.add_scatter(x=t["ym"], y=t["3-mo avg"], name="3-month avg", line=dict(color="#C0392B", width=3))
    fig.add_scatter(x=t["ym"], y=t["cumulative"], name="Cumulative", yaxis="y2", line=dict(dash="dot", color="#2C3E50"))
    fig.update_layout(title="Trend, rolling average & cumulative", yaxis2=dict(overlaying="y", side="right", title="Cumulative"))
    st.plotly_chart(fig, use_container_width=True)
    s = f.groupby(["ym", "severity_level"]).size().reset_index(name="fires")
    st.plotly_chart(px.line(s, x="ym", y="fires", color="severity_level", markers=True, color_discrete_map=SEV_COLORS,
                            title="Severity trend"), use_container_width=True)

# ---------------- Heat maps ----------------
with tabs[2]:
    hm = f.pivot_table(index="area", columns="month_name", values="incident_id", aggfunc="count", fill_value=0, observed=True)
    hm = hm.loc[hm.sum(axis=1).sort_values(ascending=False).index]
    st.plotly_chart(px.imshow(hm, text_auto=True, color_continuous_scale="Reds", aspect="auto",
                              title="Area × Month"), use_container_width=True)
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    hh = f.dropna(subset=["hour"]).pivot_table(index="day_name", columns="hour", values="incident_id",
                                               aggfunc="count", fill_value=0).reindex(days).fillna(0)
    hh = hh.reindex(columns=range(24), fill_value=0)
    st.plotly_chart(px.imshow(hh, text_auto=True, color_continuous_scale="OrRd", aspect="auto",
                              title="Day of week × Hour of call", labels=dict(x="Hour", y="")), use_container_width=True)
    zc = f.pivot_table(index="zone", columns="cause_category", values="incident_id", aggfunc="count", fill_value=0)
    st.plotly_chart(px.imshow(zc, text_auto=True, color_continuous_scale="YlOrRd", aspect="auto",
                              title="Zone × Cause"), use_container_width=True)

# ---------------- Response scorecard ----------------
with tabs[3]:
    r = f.dropna(subset=["response_minutes"])
    def card(g):
        return pd.Series({"incidents": len(g), "avg_min": g["response_minutes"].mean(),
                          "median_min": g["response_minutes"].median(), "max_min": g["response_minutes"].max(),
                          "pct_within_target": (g["response_minutes"] <= RESPONSE_TARGET_MIN).mean(),
                          "pct_out_before_crew": g["out_before_fire_crew"].mean()})
    by_m = r.groupby("ym").apply(card).reset_index()
    by_z = r.groupby("zone").apply(card).reset_index().sort_values("avg_min", ascending=False)
    by_m["status"] = np.where(by_m["pct_within_target"] >= 0.95, "🟢 On target",
                              np.where(by_m["pct_within_target"] >= 0.8, "🟠 Watch", "🔴 Below target"))
    c = st.columns(3)
    c[0].metric("Average response", f"{r.response_minutes.mean():.2f} min")
    c[1].metric("Within target", f"{(r.response_minutes <= RESPONSE_TARGET_MIN).mean():.0%}")
    c[2].metric("Slowest", f"{r.response_minutes.max():.0f} min")
    st.subheader("Monthly scorecard")
    st.dataframe(by_m.style.format({"avg_min": "{:.2f}", "median_min": "{:.1f}", "max_min": "{:.0f}",
                                    "pct_within_target": "{:.0%}", "pct_out_before_crew": "{:.0%}", "incidents": "{:.0f}"}),
                 use_container_width=True)
    st.subheader("Zone scorecard")
    st.dataframe(by_z.style.format({"avg_min": "{:.2f}", "median_min": "{:.1f}", "max_min": "{:.0f}",
                                    "pct_within_target": "{:.0%}", "pct_out_before_crew": "{:.0%}", "incidents": "{:.0f}"}),
                 use_container_width=True)
    eb = f["extinguished_by"].value_counts().reset_index(); eb.columns = ["extinguished_by", "fires"]
    st.plotly_chart(px.bar(eb, x="fires", y="extinguished_by", orientation="h", title="Who put the fire out"),
                    use_container_width=True)

# ---------------- Cause analysis ----------------
with tabs[4]:
    p = f["cause_category"].value_counts().reset_index(); p.columns = ["cause", "fires"]
    p["cum_pct"] = p["fires"].cumsum() / p["fires"].sum() * 100
    fig = go.Figure()
    fig.add_bar(x=p["cause"], y=p["fires"], name="Fires", marker_color="#C0392B")
    fig.add_scatter(x=p["cause"], y=p["cum_pct"], name="Cumulative %", yaxis="y2", mode="lines+markers")
    fig.update_layout(title="Pareto of causes", yaxis2=dict(overlaying="y", side="right", range=[0, 105], ticksuffix="%"))
    st.plotly_chart(fig, use_container_width=True)
    a, b = st.columns(2)
    a.plotly_chart(px.sunburst(f, path=["zone", "area", "cause_category"], title="Zone → Area → Cause"),
                   use_container_width=True)
    fu = f["fuel_type"].value_counts().reset_index(); fu.columns = ["fuel", "fires"]
    b.plotly_chart(px.bar(fu, x="fires", y="fuel", orientation="h", title="Material burned"), use_container_width=True)
    oa = f.groupby(["operating_activity", "severity_level"]).size().reset_index(name="fires")
    st.plotly_chart(px.bar(oa, x="operating_activity", y="fires", color="severity_level", color_discrete_map=SEV_COLORS,
                           title="Plant activity at time of fire"), use_container_width=True)

# ---------------- Equipment risk ----------------
with tabs[5]:
    er = equipment_risk(f, inc["incident_date"].max())
    if er.empty:
        st.info("No tagged equipment in the selection.")
    else:
        top = er.head(15)
        st.plotly_chart(px.bar(top[::-1], x="risk_score", y="primary_tag_base", color="risk_band", orientation="h",
                               color_discrete_map=SEV_COLORS, hover_data=["area", "fires", "last_fire"],
                               title="Equipment risk ranking (fires × severity × injuries × recency)"),
                        use_container_width=True)
        st.dataframe(er[["risk_rank", "primary_tag_base", "area", "equipment_type", "fires", "avg_severity",
                         "injuries", "first_fire", "last_fire", "days_since_last", "risk_score", "risk_band"]],
                     use_container_width=True, hide_index=True)
    et = f.groupby("equipment_type").agg(fires=("incident_id", "count"),
                                         high=("severity_level", lambda s: (s == "High").sum())).reset_index()
    st.plotly_chart(px.treemap(et, path=["equipment_type"], values="fires", color="high",
                               color_continuous_scale="Reds", title="Fires by equipment type (colour = high severity)"),
                    use_container_width=True)

# ---------------- ML ----------------
with tabs[6]:
    if "area_risk" in ml:
        ar = ml["area_risk"]
        st.plotly_chart(px.bar(ar.sort_values("risk_score_30d"), x="risk_score_30d", y="area", color="risk_band",
                               orientation="h", color_discrete_map=SEV_COLORS,
                               hover_data=["prob_ml_30d", "prob_hist_30d", "fires_ytd"],
                               title="Probability of ≥1 fire in the next 30 days"), use_container_width=True)
        st.dataframe(ar, use_container_width=True, hide_index=True)
    else:
        st.info("Run the ML notebook to create fire_ml_models/area_risk_30d.csv")
    if "metrics" in ml:
        st.subheader("Model performance"); st.dataframe(ml["metrics"], use_container_width=True, hide_index=True)

# ---------------- Reports ----------------
with tabs[7]:
    st.write("Download the filtered data and summary tables.")
    er = equipment_risk(f, inc["incident_date"].max())
    summary = pd.DataFrame({
        "KPI": ["Total fires", "Avg response (min)", "% within target", "% out before crew", "Injury incidents",
                "High severity", "Repeat-equipment fires"],
        "Value": [len(f), round(f.response_minutes.mean(), 2),
                  round((f.response_minutes.dropna() <= RESPONSE_TARGET_MIN).mean() * 100, 1),
                  round(f.out_before_fire_crew.mean() * 100, 1), int(f.injury_reported.sum()),
                  int((f.severity_level == "High").sum()), int(f.is_repeat_equipment.sum())]})
    sheets = {"KPI_Summary": summary,
              "Monthly": f.groupby(["ym", "fire_nature"]).size().unstack(fill_value=0).reset_index(),
              "By_Area": f.groupby("area").size().sort_values(ascending=False).reset_index(name="fires"),
              "By_Cause": f["cause_category"].value_counts().reset_index(),
              "Equipment_Risk": er,
              "Incidents": f.drop(columns=["ym"])}
    if "area_risk" in ml:
        sheets["Area_Risk_30d"] = ml["area_risk"]
    stamp = datetime.now().strftime("%Y%m%d")
    st.download_button("⬇️ Excel report", to_excel(sheets), f"Fire_Dashboard_Report_{stamp}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.download_button("⬇️ Filtered incidents (CSV)", f.to_csv(index=False).encode("utf-8-sig"),
                       f"fire_incidents_filtered_{stamp}.csv", "text/csv")
    if not er.empty:
        st.download_button("⬇️ Equipment risk ranking (CSV)", er.to_csv(index=False).encode("utf-8-sig"),
                           f"equipment_risk_{stamp}.csv", "text/csv")
