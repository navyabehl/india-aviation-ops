"""
ETL Pipeline — Indian Aviation Operations
Source: DGCA / Ministry of Civil Aviation (via github.com/Vonter/india-aviation-traffic)
Author: Navya Behl
"""

import pandas as pd
import numpy as np
import os

RAW = os.path.join(os.path.dirname(__file__), "../data/raw")
PROCESSED = os.path.join(os.path.dirname(__file__), "../data/processed")
os.makedirs(PROCESSED, exist_ok=True)


# ─────────────────────────────────────────────
# 1. DAILY OTP + TRAFFIC TABLE
# ─────────────────────────────────────────────
def process_daily():
    df = pd.read_csv(f"{RAW}/daily.csv", parse_dates=["Date"])

    otp_cols = [c for c in df.columns if c.startswith("On Time Performance")]
    traffic_cols = ["Date",
                    "Domestic (Departure Flights)", "Domestic (Departing Pax)",
                    "Domestic (Arrival Flights)",   "Domestic (Arriving Pax)",
                    "International (Departure Flights)", "International (Departing Pax)"]

    # OTP table
    otp = df[["Date"] + otp_cols].copy()
    otp.columns = ["Date"] + [c.replace("On Time Performance (", "").rstrip(")") for c in otp_cols]
    otp = otp.melt(id_vars="Date", var_name="Airline", value_name="OTP")
    otp = otp.dropna(subset=["OTP"])
    otp["OTP"] = pd.to_numeric(otp["OTP"], errors="coerce")
    otp["Year"]  = otp["Date"].dt.year
    otp["Month"] = otp["Date"].dt.month
    otp["MonthName"] = otp["Date"].dt.strftime("%b")
    otp.to_csv(f"{PROCESSED}/otp_daily.csv", index=False)

    # Traffic table
    traffic = df[[c for c in traffic_cols if c in df.columns]].copy()
    for col in traffic.columns[1:]:
        traffic[col] = pd.to_numeric(traffic[col], errors="coerce")
    traffic["Year"]  = traffic["Date"].dt.year
    traffic["Month"] = traffic["Date"].dt.month
    traffic.to_csv(f"{PROCESSED}/traffic_daily.csv", index=False)

    print(f"[daily] OTP rows: {len(otp):,} | Traffic rows: {len(traffic):,}")
    return otp, traffic


# ─────────────────────────────────────────────
# 2. CARRIER-LEVEL MONTHLY TABLE
# ─────────────────────────────────────────────
def process_carrier():
    df = pd.read_csv(f"{RAW}/carrier.csv")
    df.columns = df.columns.str.strip()

    # Keep only Scheduled Domestic
    df = df[df["Type"] == "ScheduledDomestic"].copy()

    numeric_cols = ["Aircraft Number", "Passenger Number", "Passenger Load Factor",
                    "Passenger Kilometers", "Seat Kilometers", "Total Cargo",
                    "Aircraft Hours", "Aircraft Kilometres"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Date"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str).str.zfill(2) + "-01"
    )
    df["MarketShare"] = df.groupby(["Year", "Month"])["Passenger Number"].transform(
        lambda x: x / x.sum() * 100
    )

    df.to_csv(f"{PROCESSED}/carrier_monthly.csv", index=False)
    print(f"[carrier] rows: {len(df):,} | airlines: {df['Airline'].nunique()}")
    return df


# ─────────────────────────────────────────────
# 3. CITY-PAIR ROUTE TABLE
# ─────────────────────────────────────────────
def process_city():
    df = pd.read_csv(f"{RAW}/city.csv")
    df.columns = df.columns.str.strip()

    for col in ["PaxToCity2", "PaxFromCity2", "FreightToCity2", "FreightFromCity2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["TotalPax"]     = df["PaxToCity2"] + df["PaxFromCity2"]
    df["TotalFreight"] = df["FreightToCity2"] + df["FreightFromCity2"]
    df["Route"]        = df["City1"] + " ↔ " + df["City2"]
    df["Date"]         = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str).str.zfill(2) + "-01"
    )

    df.to_csv(f"{PROCESSED}/city_routes.csv", index=False)
    print(f"[city] rows: {len(df):,} | unique routes: {df['Route'].nunique()}")
    return df


# ─────────────────────────────────────────────
# 4. OPERATIONAL KPI SUMMARY (single source of truth)
# ─────────────────────────────────────────────
def build_kpi_summary(carrier_df):
    kpi = carrier_df.groupby(["Year", "Month", "Airline"]).agg(
        Flights        = ("Aircraft Number", "sum"),
        Passengers     = ("Passenger Number", "sum"),
        PLF            = ("Passenger Load Factor", "mean"),
        PaxKm          = ("Passenger Kilometers", "sum"),
        MarketShare    = ("MarketShare", "mean"),
        FlightHours    = ("Aircraft Hours", "sum"),
    ).reset_index()

    kpi["Date"] = pd.to_datetime(
        kpi["Year"].astype(str) + "-" + kpi["Month"].astype(str).str.zfill(2) + "-01"
    )
    kpi["MonthName"] = kpi["Date"].dt.strftime("%b %Y")

    kpi.to_csv(f"{PROCESSED}/kpi_summary.csv", index=False)
    print(f"[kpi] summary rows: {len(kpi):,}")
    return kpi


if __name__ == "__main__":
    print("Running ETL pipeline...")
    otp, traffic = process_daily()
    carrier      = process_carrier()
    city         = process_city()
    kpi          = build_kpi_summary(carrier)
    print("\n✅ ETL complete. Processed files saved to data/processed/")
