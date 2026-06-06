"""
India Aviation Operations Performance Dashboard
Data Source: DGCA / Ministry of Civil Aviation
Author: Navya Behl | github.com/navyabehl
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os

# ─── Config ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="India Aviation Ops Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(BASE, "data", "processed")
MODELS    = os.path.join(BASE, "data", "models")

PALETTE = {
    "IndiGo":       "#2563EB",
    "Air India":    "#DC2626",
    "SpiceJet":     "#D97706",
    "Vistara":      "#7C3AED",
    "Akasa Air":    "#059669",
    "Air Asia":     "#DB2777",
    "GoAir":        "#0891B2",
    "Alliance Air": "#65A30D",
}
DEFAULT_COLOR = "#94A3B8"

# ─── Data loaders ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    kpi      = pd.read_csv(f"{PROCESSED}/kpi_summary.csv", parse_dates=["Date"])
    otp      = pd.read_csv(f"{PROCESSED}/otp_monthly.csv", parse_dates=["Date"])
    traffic  = pd.read_csv(f"{PROCESSED}/traffic_daily.csv", parse_dates=["Date"])
    routes   = pd.read_csv(f"{PROCESSED}/city_routes.csv", parse_dates=["Date"])
    plf_pred = pd.read_csv(f"{PROCESSED}/plf_predictions.csv", parse_dates=["Date"])
    try:
        fi = pd.read_csv(f"{PROCESSED}/feature_importance.csv")
    except Exception:
        fi = pd.DataFrame()
    return kpi, otp, traffic, routes, plf_pred, fi

@st.cache_resource
def load_model():
    with open(f"{MODELS}/plf_model.pkl", "rb") as f:
        return pickle.load(f)

kpi, otp, traffic, routes, plf_pred, fi = load_data()
model_bundle = load_model()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/DGCA_India_logo.svg/800px-DGCA_India_logo.svg.png", width=120)
    st.title("India Aviation Ops")
    st.caption("Data: DGCA & MoCA | 2015–2025")

    all_airlines = sorted(kpi["Airline"].dropna().unique())
    major = [a for a in ["IndiGo", "Air India", "SpiceJet", "Vistara", "Akasa Air", "Air Asia"] if a in all_airlines]
    selected_airlines = st.multiselect("Airlines", all_airlines, default=major[:5])

    years = sorted(kpi["Year"].dropna().unique().astype(int))
    year_range = st.select_slider("Year Range", options=years, value=(max(2019, min(years)), max(years)))

    st.markdown("---")
    st.markdown("**Dashboard Sections**")
    page = st.radio("", ["🏠 Overview", "📈 OTP & Reliability", "👥 Traffic & Capacity",
                          "🗺️ Route Intelligence", "🤖 Predictive Model"], label_visibility="collapsed")

# ─── Filter data ──────────────────────────────────────────────────────────────
if not selected_airlines:
    selected_airlines = major[:5]

kpi_f   = kpi[(kpi["Airline"].isin(selected_airlines)) & (kpi["Year"].between(*year_range))]
otp_f   = otp[(otp["Airline"].isin(selected_airlines)) & (otp["Year"].between(*year_range))]
plf_f   = plf_pred[(plf_pred["Airline"].isin(selected_airlines)) & (plf_pred["Year"].between(*year_range))]
traffic_f = traffic[traffic["Year"].between(*year_range)]

def airline_color(airline):
    return PALETTE.get(airline, DEFAULT_COLOR)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("✈️ India Aviation Operations Dashboard")
    st.caption(f"Showing **{', '.join(selected_airlines)}** | {year_range[0]}–{year_range[1]}")

    # KPI cards
    total_pax   = kpi_f["Passengers"].sum()
    total_flights = kpi_f["Flights"].sum()
    avg_plf     = kpi_f["PLF"].mean()
    avg_otp     = otp_f["OTP_Mean"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Passengers", f"{total_pax/1e6:.1f}M")
    c2.metric("Total Flights",    f"{total_flights/1e3:.0f}K")
    c3.metric("Avg Load Factor",  f"{avg_plf:.1f}%")
    c4.metric("Avg On-Time Perf", f"{avg_otp:.1f}%" if not np.isnan(avg_otp) else "N/A")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Monthly Passengers by Airline")
        fig = px.area(
            kpi_f.sort_values("Date"),
            x="Date", y="Passengers", color="Airline",
            color_discrete_map=PALETTE,
            labels={"Passengers": "Passengers", "Date": ""},
        )
        fig.update_layout(height=340, margin=dict(t=10, b=10), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Market Share (Latest Year)")
        latest_yr = kpi_f["Year"].max()
        ms = kpi_f[kpi_f["Year"] == latest_yr].groupby("Airline")["Passengers"].sum().reset_index()
        ms["Share"] = ms["Passengers"] / ms["Passengers"].sum() * 100
        fig = px.pie(ms, names="Airline", values="Share",
                     color="Airline", color_discrete_map=PALETTE,
                     hole=0.45)
        fig.update_traces(textposition="outside", textinfo="label+percent")
        fig.update_layout(height=340, margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("YoY Passenger Growth by Airline")
    yoy = kpi_f.groupby(["Year", "Airline"])["Passengers"].sum().reset_index()
    yoy["Growth"] = yoy.groupby("Airline")["Passengers"].pct_change() * 100
    fig = px.bar(yoy.dropna(subset=["Growth"]),
                 x="Year", y="Growth", color="Airline",
                 color_discrete_map=PALETTE, barmode="group",
                 labels={"Growth": "YoY Growth (%)", "Year": ""})
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=300, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — OTP & RELIABILITY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 OTP & Reliability":
    st.title("📈 On-Time Performance & Operational Reliability")

    if otp_f.empty:
        st.warning("No OTP data available for selected filters.")
    else:
        # OTP trend line
        st.subheader("OTP Trend Over Time")
        fig = px.line(otp_f.sort_values("Date"),
                      x="Date", y="OTP_Mean", color="Airline",
                      color_discrete_map=PALETTE,
                      labels={"OTP_Mean": "On-Time Performance (%)", "Date": ""})
        fig.add_hrect(y0=0, y1=70, fillcolor="red", opacity=0.05, annotation_text="Critical Zone")
        fig.add_hrect(y0=70, y1=80, fillcolor="orange", opacity=0.05, annotation_text="Poor Zone")
        fig.add_hrect(y0=90, y1=101, fillcolor="green", opacity=0.05, annotation_text="Excellent Zone")
        fig.update_layout(height=380, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("OTP Distribution by Airline")
            fig = px.box(otp_f, x="Airline", y="OTP_Mean", color="Airline",
                         color_discrete_map=PALETTE,
                         labels={"OTP_Mean": "OTP (%)"})
            fig.update_layout(height=340, showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("⚠️ Operational Risk Alerts (OTP < 75%)")
            alerts = otp_f[otp_f["OTP_Alert"] == True][["Date", "Airline", "OTP_Mean"]].copy()
            alerts["OTP_Mean"] = alerts["OTP_Mean"].round(1)
            alerts = alerts.sort_values("OTP_Mean").head(15)
            if alerts.empty:
                st.success("No critical OTP alerts in selected period.")
            else:
                st.dataframe(
                    alerts.rename(columns={"OTP_Mean": "OTP (%)"}),
                    use_container_width=True, hide_index=True
                )

        st.subheader("Seasonal OTP Heatmap")
        pivot = otp_f.groupby(["Month", "Airline"])["OTP_Mean"].mean().reset_index()
        pivot_wide = pivot.pivot(index="Airline", columns="Month", values="OTP_Mean")
        pivot_wide.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot_wide.columns)]
        fig = px.imshow(pivot_wide, color_continuous_scale="RdYlGn",
                        zmin=60, zmax=100,
                        labels={"color": "OTP (%)"}, aspect="auto")
        fig.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — TRAFFIC & CAPACITY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Traffic & Capacity":
    st.title("👥 Traffic, Capacity & Operational Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Passenger Load Factor Trend")
        fig = px.line(kpi_f.sort_values("Date"),
                      x="Date", y="PLF", color="Airline",
                      color_discrete_map=PALETTE,
                      labels={"PLF": "Load Factor (%)", "Date": ""})
        fig.add_hline(y=80, line_dash="dot", line_color="gray",
                      annotation_text="Industry benchmark 80%")
        fig.update_layout(height=340, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Flights Operated Monthly")
        fig = px.bar(kpi_f.sort_values("Date"),
                     x="Date", y="Flights", color="Airline",
                     color_discrete_map=PALETTE, barmode="stack",
                     labels={"Flights": "Aircraft Operated", "Date": ""})
        fig.update_layout(height=340, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Daily Domestic Traffic — Flight Movements vs Passengers")
    if not traffic_f.empty:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        agg = traffic_f.groupby("Date").agg({
            "Domestic (Departure Flights)": "sum",
            "Domestic (Departing Pax)": "sum"
        }).reset_index().dropna()
        fig.add_trace(go.Scatter(x=agg["Date"], y=agg["Domestic (Departure Flights)"],
                                 name="Flights", line=dict(color="#2563EB")), secondary_y=False)
        fig.add_trace(go.Scatter(x=agg["Date"], y=agg["Domestic (Departing Pax)"],
                                 name="Passengers", line=dict(color="#DC2626", dash="dot")), secondary_y=True)
        fig.update_yaxes(title_text="Flights", secondary_y=False)
        fig.update_yaxes(title_text="Passengers", secondary_y=True)
        fig.update_layout(height=340, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Airline Efficiency: Passengers per Flight")
    eff = kpi_f.copy()
    eff["PaxPerFlight"] = eff["Passengers"] / eff["Flights"].replace(0, np.nan)
    eff_agg = eff.groupby(["Year", "Airline"])["PaxPerFlight"].mean().reset_index()
    fig = px.line(eff_agg, x="Year", y="PaxPerFlight", color="Airline",
                  color_discrete_map=PALETTE, markers=True,
                  labels={"PaxPerFlight": "Avg Pax per Flight"})
    fig.update_layout(height=300, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ROUTE INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Route Intelligence":
    st.title("🗺️ Route Intelligence & Network Analysis")

    routes_f = routes[routes["Year"].between(*year_range)]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 20 Busiest Routes")
        top_routes = (routes_f.groupby("Route")["TotalPax"]
                      .sum().reset_index()
                      .sort_values("TotalPax", ascending=False).head(20))
        fig = px.bar(top_routes, x="TotalPax", y="Route", orientation="h",
                     color="TotalPax", color_continuous_scale="Blues",
                     labels={"TotalPax": "Total Passengers", "Route": ""})
        fig.update_layout(height=500, margin=dict(t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top Origin Cities by Passenger Volume")
        city_vol = (routes_f.groupby("City1")["TotalPax"]
                    .sum().reset_index()
                    .sort_values("TotalPax", ascending=False).head(15))
        fig = px.bar(city_vol, x="TotalPax", y="City1", orientation="h",
                     color="TotalPax", color_continuous_scale="Purples",
                     labels={"TotalPax": "Total Passengers", "City1": "City"})
        fig.update_layout(height=500, margin=dict(t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Route Traffic Trend — Search a Route")
    all_routes = sorted(routes_f["Route"].unique())
    popular_defaults = ["DELHI ↔ MUMBAI", "DELHI ↔ BANGALORE", "MUMBAI ↔ BANGALORE"]
    default_sel = [r for r in popular_defaults if r in all_routes][:2]
    sel_routes  = st.multiselect("Select Routes", all_routes, default=default_sel)

    if sel_routes:
        route_trend = routes_f[routes_f["Route"].isin(sel_routes)].groupby(
            ["Date", "Route"])["TotalPax"].sum().reset_index()
        fig = px.line(route_trend, x="Date", y="TotalPax", color="Route",
                      labels={"TotalPax": "Passengers", "Date": ""})
        fig.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Freight Concentration by Route")
    freight_routes = (routes_f.groupby("Route")["TotalFreight"]
                      .sum().reset_index()
                      .sort_values("TotalFreight", ascending=False).head(15))
    fig = px.treemap(freight_routes, path=["Route"], values="TotalFreight",
                     color="TotalFreight", color_continuous_scale="Oranges",
                     labels={"TotalFreight": "Freight (MT)"})
    fig.update_layout(height=380, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PREDICTIVE MODEL
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Predictive Model":
    st.title("🤖 Predictive Model — Passenger Load Factor")
    st.caption(f"Best model: **{model_bundle['best_name']}** | Trained on DGCA carrier data 2015–2024")

    # Model performance cards
    st.subheader("Model Performance Comparison")
    res = model_bundle["results"]
    mc = st.columns(len(res))
    for i, (name, m) in enumerate(res.items()):
        mc[i].metric(name, f"R² = {m['R2']:.3f}", f"MAE = {m['MAE']:.1f}pp")

    col1, col2 = st.columns(2)

    with col1:
        # Actual vs Predicted scatter
        st.subheader("Actual vs Predicted PLF")
        fig = px.scatter(plf_f, x="PLF", y="PLF_Predicted", color="Airline",
                         color_discrete_map=PALETTE,
                         opacity=0.7,
                         labels={"PLF": "Actual PLF (%)", "PLF_Predicted": "Predicted PLF (%)"})
        fig.add_shape(type="line", x0=plf_f["PLF"].min(), y0=plf_f["PLF"].min(),
                      x1=plf_f["PLF"].max(), y1=plf_f["PLF"].max(),
                      line=dict(color="black", dash="dash"))
        fig.update_layout(height=380, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Feature importance
        if not fi.empty:
            st.subheader("Feature Importance")
            fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="Blues")
            fig.update_layout(height=380, margin=dict(t=10, b=10),
                              coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("🎯 Live PLF Predictor")
    st.caption("Simulate load factor for an airline under given operating conditions")

    encoder = model_bundle["encoder"]
    available_airlines = list(encoder.classes_)

    pc1, pc2, pc3, pc4 = st.columns(4)
    pred_airline  = pc1.selectbox("Airline", available_airlines,
                                  index=available_airlines.index("IndiGo") if "IndiGo" in available_airlines else 0)
    pred_month    = pc2.slider("Month", 1, 12, 6)
    pred_flights  = pc3.number_input("Flights operated", min_value=1, max_value=5000, value=500)
    pred_mktshare = pc4.number_input("Market share (%)", min_value=0.0, max_value=100.0, value=60.0)

    pred_quarter = ((pred_month - 1) // 3) + 1
    pred_ispeak  = int(pred_month in [1, 3, 5, 10, 11, 12])
    pred_hours   = pred_flights * 1.8  # avg flight hours proxy

    input_df = pd.DataFrame([{
        "AirlineCode":  encoder.transform([pred_airline])[0],
        "Month":        pred_month,
        "Quarter":      pred_quarter,
        "IsPeak":       pred_ispeak,
        "Flights":      pred_flights,
        "MarketShare":  pred_mktshare,
        "FlightHours":  pred_hours,
    }])

    predicted_plf = model_bundle["model"].predict(input_df)[0]
    plf_color = "green" if predicted_plf >= 80 else ("orange" if predicted_plf >= 70 else "red")

    st.markdown(f"""
    <div style='background:#1e293b;padding:24px;border-radius:12px;text-align:center;margin-top:12px'>
        <p style='color:#94a3b8;font-size:14px;margin:0'>Predicted Passenger Load Factor</p>
        <p style='color:{plf_color};font-size:52px;font-weight:700;margin:8px 0'>{predicted_plf:.1f}%</p>
        <p style='color:#64748b;font-size:13px'>{'✅ Above industry benchmark (80%)' if predicted_plf >= 80 else '⚠️ Below industry benchmark (80%)'}</p>
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Data: DGCA & Ministry of Civil Aviation via [Vonter/india-aviation-traffic](https://github.com/Vonter/india-aviation-traffic) | Built by Navya Behl")
