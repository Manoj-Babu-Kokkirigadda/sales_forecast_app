# 📈 Sales Forecast Pro

**AI-Powered Sales Prediction & Forecasting Application**  
IBM University Engagement Program — 2025  
**Author:** KOKKIRIGADDA MANOJ BABU  
**Email:** kokkirigaddamanojbabu@gmail.com | **WhatsApp:** 9553848655

---

## 🚀 Project Overview

**Sales Forecast Pro** is a full-featured, interactive web application built with **Streamlit** that enables businesses to analyze historical sales data and predict future revenues using **8 machine learning and deep learning algorithms**. The application covers the complete ML lifecycle — from data exploration and feature engineering to model training, evaluation, comparison, and multi-horizon forecasting.

The project uses the **Superstore Sales Dataset** (9,800 transactions from 2015–2018) and achieves its best performance with a **Random Forest** model (MAE: 1,796 | R²: 0.185).

---

## ✨ Features

### 🏠 Dashboard
- **KPI Cards:** Total Revenue ($2.26M), Avg Order Value, Total Orders (4,922), Avg Daily Sales
- **Daily Sales Trend** — interactive line chart over the full 4-year period
- **Sales by Category** — donut chart (Furniture, Office Supplies, Technology)
- **Monthly Sales Revenue** — bar chart with monthly granularity
- **Sales by Customer Segment** — Consumer, Corporate, Home Office
- **Year-over-Year Comparison** — multi-line chart (2015–2018)

### 📊 Data Explorer
- **Raw Data** tab with filterable, sortable dataset preview
- **Distributions** tab — histograms and box plots for sales analysis
- **Correlations** tab — heatmaps across numerical features
- **Custom Aggregation** tab — aggregate sales by Region, Segment, Category, Sub-Category, or State

### 🎯 Predict Sale (Single Date)
- Pick any future date → instantly get predicted daily sales
- Uses the **best-trained model** (auto-selected after training)
- Shows prediction confidence range and historical comparison

### 🤖 Model Training
- Train all **8 algorithms** with a single click
- Algorithms: ARIMA, Prophet, Ridge Regression, Random Forest, Gradient Boosting, XGBoost, LSTM, GRU
- Full metrics displayed: **MAE, RMSE, R², SMAPE, MAPE**
- Auto-crowns the best model and saves it for future predictions

### 🔮 Forecast (Multi-Horizon)
- **30 / 60 / 90-day** rolling forecast with confidence intervals
- Interactive forecast chart with seasonal trend overlay
- Downloadable forecast table (CSV)

### 📈 Model Comparison
- Side-by-side leaderboard of all 8 models
- MAE and R² bar charts — best model highlighted in **green**
- Detailed metrics table for all algorithms

---

## 🏆 Model Performance Results

| Model | MAE | RMSE | R² | SMAPE |
|-------|-----|------|----|-------|
| **Random Forest** ✅ | **1,797** | **2,274** | **0.185** | 66.3% |
| XGBoost | 1,782 | 2,274 | 0.185 | 65.5% |
| Ridge Regression | 1,886 | 2,297 | 0.169 | 68.9% |
| Gradient Boosting | 2,045 | 2,493 | 0.020 | 70.8% |
| ARIMA | 2,207 | 3,214 | -0.186 | 82.9% |
| Prophet | 2,214 | 3,068 | -0.081 | 79.9% |
| GRU | 2,525 | 3,635 | -0.518 | 88.2% |
| LSTM | 3,344 | 4,286 | -1.110 | 100.3% |

> ✅ **Best Model: Random Forest** — lowest MAE, highest R²

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.x** | Core programming language |
| **Streamlit** | Web application framework & UI |
| **Pandas / NumPy** | Data manipulation and numerical computation |
| **Scikit-learn** | Ridge Regression, Random Forest, Gradient Boosting |
| **XGBoost** | Extreme Gradient Boosting |
| **TensorFlow / Keras** | LSTM and GRU deep learning models |
| **Prophet (Meta)** | Time-series decomposition & seasonal forecasting |
| **Statsmodels (ARIMA)** | Classical statistical time-series model |
| **Plotly** | Interactive charts and visualizations |
| **Joblib** | Model serialization and caching |

---

## 📁 Repository Structure

```
sales_forecast_app/
├── app.py                          # Main Streamlit application (6 pages)
├── train_models.py                 # Model training pipeline (8 algorithms)
├── utils.py                        # Data loading, feature engineering helpers
├── requirements.txt                # Python dependencies
├── 3 Sales Forecast Prediction.csv # Superstore dataset (9,800 rows)
├── models/
│   ├── best_model.pkl              # Serialized best model (Random Forest)
│   ├── best_model_info.json        # Best model name and metrics
│   └── model_results.json          # All 8 model results and metrics
└── README.md                       # This file
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/Manoj-Babu-Kokkirigadda/sales_forecast_app.git
cd sales_forecast_app
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Train the Models
> Skip this step if the `models/` folder already contains `best_model.pkl`
```bash
python train_models.py
```
This trains all 8 models and saves the best one. Takes ~2–5 minutes.

### Step 4: Launch the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`

---

## 📦 Requirements

```
streamlit
pandas
numpy
plotly
scikit-learn
xgboost
prophet
statsmodels
tensorflow
keras
joblib
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

**Superstore Sales Dataset**
- **Source:** Sample retail sales data (commonly used in BI/analytics)
- **File:** `3 Sales Forecast Prediction.csv`
- **Rows:** 9,800 transactions
- **Period:** January 2015 – December 2018
- **Features:** Order ID, Order Date, Ship Date, Customer Segment, Region, Category, Sub-Category, Product Name, Sales

---

## 🔧 Feature Engineering

The following time-series features are extracted from raw order dates for ML models:

| Feature | Description |
|---------|-------------|
| `lag_1` to `lag_7` | Sales values from previous 1–7 days |
| `rolling_mean_7` | 7-day rolling average |
| `rolling_mean_14` | 14-day rolling average |
| `rolling_std_7` | 7-day rolling standard deviation |
| `day_of_week` | Day of week (0–6) |
| `month` | Month number (1–12) |
| `quarter` | Quarter (1–4) |
| `year` | Calendar year |
| `day_of_year` | Day of year (1–365) |

---

## 🔮 Future Scope

1. **Real-Time Data Integration** — Connect to live POS/ERP systems for continuous retraining
2. **Advanced Deep Learning** — Transformer models (TFT, N-BEATS, PatchTST)
3. **SKU-Level Forecasting** — Granular product and inventory-level predictions
4. **Automated Alerts** — Email/SMS notifications for forecast deviations
5. **IBM watsonx.ai Integration** — Enterprise-grade AI deployment with governance
6. **Multi-Store Support** — Hierarchical forecasting across stores and regions

---

## 📄 License

This project was developed as part of the **IBM University Engagement Program 2025** for academic/internship purposes.

---

## 👤 Author

**KOKKIRIGADDA MANOJ BABU**  
📧 kokkirigaddamanojbabu@gmail.com  
📱 WhatsApp: 9553848655  
🔗 [GitHub Repository](https://github.com/Manoj-Babu-Kokkirigadda/sales_forecast_app)

---

*IBM University Engagement Program — Project Submission 2025*
