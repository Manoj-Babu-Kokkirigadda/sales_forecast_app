import pandas as pd
import numpy as np
from datetime import datetime
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "3 Sales Forecast Prediction.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")
    return df


def get_daily_sales(df):
    daily = df.groupby("Order Date")["Sales"].sum().reset_index()
    daily.columns = ["ds", "y"]
    daily = daily.sort_values("ds").reset_index(drop=True)
    daily["y"] = daily["y"].round(2)
    return daily


def get_monthly_sales(df):
    monthly = df.copy()
    monthly["ym"] = monthly["Order Date"].dt.to_period("M")
    monthly = monthly.groupby("ym")["Sales"].sum().reset_index()
    monthly["ym"] = monthly["ym"].astype(str)
    monthly.columns = ["Month", "Sales"]
    return monthly


def get_weekly_sales(df):
    weekly = df.copy()
    weekly["week"] = weekly["Order Date"].dt.to_period("W")
    weekly = weekly.groupby("week")["Sales"].sum().reset_index()
    weekly["week"] = weekly["week"].astype(str)
    weekly.columns = ["Week", "Sales"]
    return weekly


def get_yearly_sales(df):
    yearly = df.copy()
    yearly["Year"] = yearly["Order Date"].dt.year
    yearly = yearly.groupby("Year")["Sales"].sum().reset_index()
    return yearly


def get_segment_sales(df):
    return df.groupby("Segment")["Sales"].sum().reset_index()


def get_category_sales(df):
    return df.groupby("Category")["Sales"].sum().reset_index()


def get_region_sales(df):
    return df.groupby("Region")["Sales"].sum().reset_index()


def get_subcategory_sales(df):
    return df.groupby("Sub-Category")["Sales"].sum().reset_index().sort_values("Sales", ascending=False)


def get_daily_features_from_full(df):
    df_feat = df.copy()
    df_feat["Order Date"] = pd.to_datetime(df_feat["Order Date"], format="%d/%m/%Y")
    features = df_feat.groupby("Order Date").agg(
        num_orders=("Order ID", "nunique"),
        num_products=("Product ID", "nunique"),
        num_customers=("Customer ID", "nunique"),
        avg_sale=("Sales", "mean"),
        max_sale=("Sales", "max"),
        min_sale=("Sales", "min"),
        std_sale=("Sales", "std"),
        num_segments=("Segment", "nunique"),
        num_categories=("Category", "nunique"),
        num_regions=("Region", "nunique"),
    ).reset_index()
    features = features.fillna(0)
    features.columns = ["ds"] + [f"daily_{c}" for c in features.columns if c != "Order Date"]
    return features


def create_features(df):
    df_feat = df.copy()
    df_feat["year"] = df_feat["ds"].dt.year
    df_feat["month"] = df_feat["ds"].dt.month
    df_feat["day"] = df_feat["ds"].dt.day
    df_feat["dayofweek"] = df_feat["ds"].dt.dayofweek
    df_feat["quarter"] = df_feat["ds"].dt.quarter
    df_feat["dayofyear"] = df_feat["ds"].dt.dayofyear
    df_feat["weekofyear"] = df_feat["ds"].dt.isocalendar().week.astype(int)
    df_feat["is_weekend"] = (df_feat["dayofweek"] >= 5).astype(int)
    df_feat["is_month_start"] = (df_feat["ds"].dt.is_month_start).astype(int)
    df_feat["is_month_end"] = (df_feat["ds"].dt.is_month_end).astype(int)
    df_feat["days_in_month"] = df_feat["ds"].dt.days_in_month
    df_feat["day_of_month_ratio"] = df_feat["day"] / df_feat["days_in_month"]
    for lag in [1, 2, 3, 7, 14, 21, 28, 60]:
        df_feat[f"lag_{lag}"] = df_feat["y"].shift(lag)
    for w in [7, 14, 28, 60]:
        df_feat[f"rolling_mean_{w}"] = df_feat["y"].rolling(window=w).mean()
        df_feat[f"rolling_std_{w}"] = df_feat["y"].rolling(window=w).std()
        df_feat[f"rolling_max_{w}"] = df_feat["y"].rolling(window=w).max()
        df_feat[f"rolling_min_{w}"] = df_feat["y"].rolling(window=w).min()
    df_feat["month_sin"] = np.sin(2 * np.pi * df_feat["month"] / 12)
    df_feat["month_cos"] = np.cos(2 * np.pi * df_feat["month"] / 12)
    df_feat["day_sin"] = np.sin(2 * np.pi * df_feat["dayofweek"] / 7)
    df_feat["day_cos"] = np.cos(2 * np.pi * df_feat["dayofweek"] / 7)
    df_feat["dayofyear_sin"] = np.sin(2 * np.pi * df_feat["dayofyear"] / 365.25)
    df_feat["dayofyear_cos"] = np.cos(2 * np.pi * df_feat["dayofyear"] / 365.25)
    df_feat["week_sin"] = np.sin(2 * np.pi * df_feat["weekofyear"] / 52)
    df_feat["week_cos"] = np.cos(2 * np.pi * df_feat["weekofyear"] / 52)
    return df_feat.dropna().reset_index(drop=True)
