# ✈️ India Aviation Operations Performance Dashboard

An end-to-end data analytics project analysing Indian domestic aviation performance using official DGCA and Ministry of Civil Aviation data.

Built to demonstrate **operational analytics, ETL pipeline development, predictive modelling, and executive-level data visualisation** — the core skillset for aviation and transport analytics roles.

**[🚀 Live App →](https://navyabehl-india-aviation-ops.streamlit.app)**

---

## 📊 What This Dashboard Does

| Module | Description |
|---|---|
| **Overview** | Traffic volumes, market share, YoY passenger growth by airline |
| **OTP & Reliability** | On-time performance trends, seasonal heatmaps, operational risk alerts |
| **Traffic & Capacity** | Load factor trends, flight movements, efficiency metrics |
| **Route Intelligence** | Busiest routes, origin city rankings, freight concentration, route search |
| **Predictive Model** | Random Forest PLF predictor (R² = 0.868) with live simulator |

---

## 🗂️ Project Structure

```
india-aviation-ops/
│
├── app.py                  # Streamlit dashboard (5 pages)
├── setup.py                # One-time setup: download data, run ETL, train model
├── requirements.txt
│
├── src/
│   ├── etl.py              # ETL pipeline: raw → processed CSVs
│   └── model.py            # Predictive modelling: PLF forecasting + OTP trends
│
├── data/
│   ├── raw/                # Source CSVs from DGCA/MoCA (auto-downloaded)
│   └── processed/          # Cleaned, structured outputs (single source of truth)
│
└── models/
    └── plf_model.pkl       # Trained Random Forest model
```

---

## ⚙️ Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/navyabehl/india-aviation-ops.git
cd india-aviation-ops

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data, run ETL, train models (one-time)
python setup.py

# 4. Launch dashboard
streamlit run app.py
```

---

## 📦 Data Sources

| Dataset | Source | Rows | Coverage |
|---|---|---|---|
| Daily OTP & Traffic | Ministry of Civil Aviation | ~1,244 | 2022–2025 |
| Monthly Carrier Stats | DGCA Domestic Transport | ~1,889 | 2015–2024 |
| City-Pair Routes | DGCA Domestic Transport | ~64,511 | 2015–2024 |

All data sourced from official Indian government publications via [Vonter/india-aviation-traffic](https://github.com/Vonter/india-aviation-traffic).

---

## 🤖 Predictive Model

**Target:** Passenger Load Factor (PLF) — key operational efficiency metric

| Model | R² | MAE |
|---|---|---|
| **Random Forest** ✅ | **0.868** | **4.5 pp** |
| Gradient Boosting | 0.851 | 5.8 pp |
| Linear Regression | 0.081 | 16.7 pp |

**Top features:** Market share, flight frequency, seasonality (peak/off-peak), airline identity

The live simulator lets you input operating conditions and get an instant PLF prediction.

---

## 🔑 Key Operational Metrics Designed

- **On-Time Performance (OTP)** — tracked daily and monthly per airline with RAG status thresholds
- **Passenger Load Factor (PLF)** — actual vs predicted, benchmarked at 80% industry standard
- **Market Concentration** — airline-wise share of domestic passengers per period
- **Operational Risk Alerts** — automated flagging of OTP < 75% as critical events
- **Route Efficiency** — passengers per flight, freight concentration by corridor
- **Seasonal Patterns** — monthly heatmap of reliability by airline

---

## 🛠️ Tech Stack

`Python` · `Pandas` · `Scikit-learn` · `Plotly` · `Streamlit` · `SQL-style aggregations`

---

## 👤 Author

**Navya Behl** | M.Sc. Economics, Dr. B.R. Ambedkar School of Economics University (2026)

[LinkedIn](https://www.linkedin.com/in/navyabehl/) · [GitHub](https://github.com/navyabehl)
