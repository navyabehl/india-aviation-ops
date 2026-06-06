"""
Predictive Modelling — Passenger Load Factor & OTP Forecasting
Author: Navya Behl
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import os
import pickle

PROCESSED = os.path.join(os.path.dirname(__file__), "../data/processed")


def build_plf_model():
    """Predict Passenger Load Factor from airline/season/route features."""
    df = pd.read_csv(f"{PROCESSED}/kpi_summary.csv")
    df = df.dropna(subset=["PLF", "Flights", "Passengers", "MarketShare"])

    # Feature engineering
    le = LabelEncoder()
    df["AirlineCode"] = le.fit_transform(df["Airline"])
    df["Quarter"]     = ((df["Month"] - 1) // 3) + 1
    df["IsPeak"]      = df["Month"].isin([1, 3, 5, 10, 11, 12]).astype(int)  # peak travel months

    features = ["AirlineCode", "Month", "Quarter", "IsPeak", "Flights",
                "MarketShare", "FlightHours"]
    target   = "PLF"

    X = df[features].fillna(0)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Random Forest":       RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting":   GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Linear Regression":   LinearRegression(),
    }

    results = {}
    best_model, best_r2, best_name = None, -999, ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae   = mean_absolute_error(y_test, preds)
        r2    = r2_score(y_test, preds)
        results[name] = {"MAE": round(mae, 3), "R2": round(r2, 3)}
        if r2 > best_r2:
            best_r2, best_model, best_name = r2, model, name

    print(f"\nModel Performance (PLF Prediction):")
    for name, m in results.items():
        print(f"  {name:25s} → MAE: {m['MAE']:.3f} | R²: {m['R2']:.3f}")
    print(f"\n  Best model: {best_name} (R² = {best_r2:.3f})")

    # Save model + encoder + feature list
    os.makedirs(f"{PROCESSED}/../models", exist_ok=True)
    with open(f"{PROCESSED}/../models/plf_model.pkl", "wb") as f:
        pickle.dump({"model": best_model, "encoder": le, "features": features,
                     "results": results, "best_name": best_name}, f)

    # Save feature importance
    if hasattr(best_model, "feature_importances_"):
        fi = pd.DataFrame({
            "Feature":   features,
            "Importance": best_model.feature_importances_
        }).sort_values("Importance", ascending=False)
        fi.to_csv(f"{PROCESSED}/feature_importance.csv", index=False)

    # Save predictions for dashboard
    df["PLF_Predicted"] = best_model.predict(X)
    df["PLF_Residual"]  = df["PLF"] - df["PLF_Predicted"]
    df.to_csv(f"{PROCESSED}/plf_predictions.csv", index=False)

    return results


def build_otp_trends():
    """Compute OTP trend metrics for dashboard."""
    otp = pd.read_csv(f"{PROCESSED}/otp_daily.csv", parse_dates=["Date"])

    monthly_otp = otp.groupby(["Year", "Month", "Airline"]).agg(
        OTP_Mean = ("OTP", "mean"),
        OTP_Min  = ("OTP", "min"),
        OTP_Max  = ("OTP", "max"),
        OTP_Std  = ("OTP", "std"),
    ).reset_index()

    monthly_otp["Date"] = pd.to_datetime(
        monthly_otp["Year"].astype(str) + "-" +
        monthly_otp["Month"].astype(str).str.zfill(2) + "-01"
    )

    # Flag low-OTP months as operational risk alerts
    monthly_otp["OTP_Alert"] = monthly_otp["OTP_Mean"] < 75
    monthly_otp["OTP_Status"] = pd.cut(
        monthly_otp["OTP_Mean"],
        bins=[0, 70, 80, 90, 101],
        labels=["Critical (<70%)", "Poor (70–80%)", "Good (80–90%)", "Excellent (>90%)"]
    )

    monthly_otp.to_csv(f"{PROCESSED}/otp_monthly.csv", index=False)
    print(f"[OTP trends] {len(monthly_otp):,} airline-month records")
    return monthly_otp


if __name__ == "__main__":
    print("Building predictive models...")
    build_plf_model()
    build_otp_trends()
    print("\n✅ Models complete.")
