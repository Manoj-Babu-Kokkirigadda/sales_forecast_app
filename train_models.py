import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
import json
import joblib
from datetime import datetime, timedelta

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from prophet import Prophet

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from utils import load_data, get_daily_sales, create_features, get_daily_features_from_full

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

RESULTS_FILE = os.path.join(MODEL_DIR, "model_results.json")


def check_stationarity(series):
    result = adfuller(series.dropna())
    return result[1] < 0.05


def train_test_split(daily, test_size=90):
    train = daily.iloc[:-test_size].copy()
    test = daily.iloc[-test_size:].copy()
    return train, test


def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))


def calc_metrics(y_true, y_pred):
    mask = ~np.isnan(y_pred)
    if mask.sum() == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "SMAPE": np.nan, "MAPE": np.nan}
    y_true, y_pred = y_true[mask], y_pred[mask]
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    smape_val = smape(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "SMAPE": smape_val, "MAPE": mape}


def train_arima(train, test):
    try:
        d = 0 if check_stationarity(train["y"]) else 1
        model = ARIMA(train["y"], order=(3, d, 3))
        fitted = model.fit()
        forecast = fitted.forecast(steps=len(test))
        preds = forecast.values
        return {"model": fitted, "y_true": test["y"].values, "y_pred": preds, "metrics": calc_metrics(test["y"].values, preds)}
    except Exception as e:
        return {"model": None, "y_true": test["y"].values, "y_pred": np.full(len(test), np.nan), "metrics": calc_metrics(test["y"].values, np.full(len(test), np.nan))}


def train_prophet(train, test):
    try:
        prophet_df = train[["ds", "y"]].rename(columns={"ds": "ds", "y": "y"})
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                    changepoint_prior_scale=0.05, seasonality_prior_scale=10)
        m.fit(prophet_df)
        future = m.make_future_dataframe(periods=len(test))
        forecast = m.predict(future)
        preds = forecast["yhat"].iloc[-len(test):].values
        return {"model": m, "y_true": test["y"].values, "y_pred": preds, "metrics": calc_metrics(test["y"].values, preds)}
    except:
        return {"model": None, "y_true": test["y"].values, "y_pred": np.full(len(test), np.nan), "metrics": calc_metrics(test["y"].values, np.full(len(test), np.nan))}


def train_ridge(train, test):
    try:
        train_feat = create_features(train)
        test_feat = create_features(test)
        feature_cols = [c for c in train_feat.columns if c not in ["ds", "y"]]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_feat[feature_cols].values)
        y_train = train_feat["y"].values
        X_test = scaler.transform(test_feat[feature_cols].values)
        y_test = test_feat["y"].values
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return {"model": model, "scaler": scaler, "feature_cols": feature_cols, "y_true": y_test, "y_pred": preds, "metrics": calc_metrics(y_test, preds)}
    except:
        return {"model": None, "y_true": test["y"].values, "y_pred": np.full(len(test), np.nan), "metrics": calc_metrics(test["y"].values, np.full(len(test), np.nan))}


def train_random_forest(train, test):
    try:
        train_feat = create_features(train)
        test_feat = create_features(test)
        feature_cols = [c for c in train_feat.columns if c not in ["ds", "y"]]
        X_train = train_feat[feature_cols].values
        y_train = train_feat["y"].values
        X_test = test_feat[feature_cols].values
        y_test = test_feat["y"].values
        model = RandomForestRegressor(n_estimators=500, max_depth=20, min_samples_leaf=2,
                                       min_samples_split=5, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return {"model": model, "scaler": None, "feature_cols": feature_cols, "y_true": y_test, "y_pred": preds, "metrics": calc_metrics(y_test, preds)}
    except:
        return {"model": None, "y_true": test["y"].values, "y_pred": np.full(len(test), np.nan), "metrics": calc_metrics(test["y"].values, np.full(len(test), np.nan))}


def train_gradient_boosting(train, test):
    try:
        train_feat = create_features(train)
        test_feat = create_features(test)
        feature_cols = [c for c in train_feat.columns if c not in ["ds", "y"]]
        X_train = train_feat[feature_cols].values
        y_train = train_feat["y"].values
        X_test = test_feat[feature_cols].values
        y_test = test_feat["y"].values
        model = GradientBoostingRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                                           subsample=0.8, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return {"model": model, "scaler": None, "feature_cols": feature_cols, "y_true": y_test, "y_pred": preds, "metrics": calc_metrics(y_test, preds)}
    except:
        return {"model": None, "y_true": test["y"].values, "y_pred": np.full(len(test), np.nan), "metrics": calc_metrics(test["y"].values, np.full(len(test), np.nan))}


def train_xgboost(train, test):
    try:
        train_feat = create_features(train)
        test_feat = create_features(test)
        feature_cols = [c for c in train_feat.columns if c not in ["ds", "y"]]
        X_train = train_feat[feature_cols].values
        y_train = train_feat["y"].values
        X_test = test_feat[feature_cols].values
        y_test = test_feat["y"].values
        model = xgb.XGBRegressor(
            n_estimators=500, max_depth=10, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        preds = model.predict(X_test)
        return {"model": model, "scaler": None, "feature_cols": feature_cols, "y_true": y_test, "y_pred": preds, "metrics": calc_metrics(y_test, preds)}
    except Exception as e:
        return {"model": None, "y_true": test["y"].values, "y_pred": np.full(len(test), np.nan), "metrics": calc_metrics(test["y"].values, np.full(len(test), np.nan))}


def prepare_lstm_data(train, test, lookback=30):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train[["y"]].values)
    test_scaled = scaler.transform(test[["y"]].values)
    X_train, y_train = [], []
    for i in range(lookback, len(train_scaled)):
        X_train.append(train_scaled[i - lookback : i, 0])
        y_train.append(train_scaled[i, 0])
    full = np.vstack([train_scaled, test_scaled])
    X_test, y_test = [], []
    for i in range(len(train_scaled), len(full)):
        X_test.append(full[i - lookback : i, 0])
        y_test.append(full[i, 0])
    return np.array(X_train).reshape(-1, lookback, 1), np.array(y_train), np.array(X_test).reshape(-1, lookback, 1), np.array(y_test), scaler


def build_lstm_model(lookback):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(lookback, 1)),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model


def train_lstm(train, test):
    try:
        lookback = 30
        if len(train) <= lookback + 10:
            return {"model": None, "y_true": test["y"].values, "y_pred": np.full(len(test), np.nan), "metrics": calc_metrics(test["y"].values, np.full(len(test), np.nan))}
        X_train, y_train, X_test, y_test, scaler = prepare_lstm_data(train, test, lookback)
        model = build_lstm_model(lookback)
        callbacks = [
            EarlyStopping(monitor="loss", patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor="loss", factor=0.5, patience=5, min_lr=1e-6)
        ]
        model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=0, callbacks=callbacks)
        preds_scaled = model.predict(X_test, verbose=0).flatten()
        preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        return {"model": model, "scaler": scaler, "lookback": lookback, "y_true": test["y"].values, "y_pred": preds, "metrics": calc_metrics(test["y"].values, preds)}
    except:
        return {"model": None, "y_true": test["y"].values, "y_pred": np.full(len(test), np.nan), "metrics": calc_metrics(test["y"].values, np.full(len(test), np.nan))}


def train_gru(train, test):
    try:
        lookback = 30
        if len(train) <= lookback + 10:
            return {"model": None, "y_true": test["y"].values, "y_pred": np.full(len(test), np.nan), "metrics": calc_metrics(test["y"].values, np.full(len(test), np.nan))}
        X_train, y_train, X_test, y_test, scaler = prepare_lstm_data(train, test, lookback)
        model = Sequential([
            GRU(64, return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.3),
            GRU(32, return_sequences=False),
            Dropout(0.3),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
        callbacks = [EarlyStopping(monitor="loss", patience=15, restore_best_weights=True)]
        model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=0, callbacks=callbacks)
        preds_scaled = model.predict(X_test, verbose=0).flatten()
        preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        return {"model": model, "scaler": scaler, "lookback": lookback, "y_true": test["y"].values, "y_pred": preds, "metrics": calc_metrics(test["y"].values, preds)}
    except:
        return {"model": None, "y_true": test["y"].values, "y_pred": np.full(len(test), np.nan), "metrics": calc_metrics(test["y"].values, np.full(len(test), np.nan))}


def get_feature_importance(model_dict):
    if model_dict.get("model") is None:
        return []
    model = model_dict["model"]
    feature_cols = model_dict.get("feature_cols", [])
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        return sorted(zip(feature_cols, imp), key=lambda x: x[1], reverse=True)
    if hasattr(model, "coef_"):
        imp = np.abs(model.coef_)
        return sorted(zip(feature_cols, imp), key=lambda x: x[1], reverse=True)
    return []


def train_all(test_size=90):
    print("=" * 60)
    print("SALES FORECASTING - MODEL TRAINING PIPELINE")
    print("=" * 60)

    df = load_data()
    daily = get_daily_sales(df)
    train, test = train_test_split(daily, test_size)
    print(f"\nTraining data: {train['ds'].min()} to {train['ds'].max()} ({len(train)} days)")
    print(f"Test data: {test['ds'].min()} to {test['ds'].max()} ({len(test)} days)")
    print(f"\n{'=' * 60}")
    print(f"{'Model':<25} {'MAE':<12} {'RMSE':<12} {'R2':<10} {'SMAPE':<10}")
    print(f"{'=' * 60}")

    models = {
        "ARIMA": train_arima,
        "Prophet": train_prophet,
        "Ridge Regression": train_ridge,
        "Random Forest": train_random_forest,
        "Gradient Boosting": train_gradient_boosting,
        "XGBoost": train_xgboost,
        "LSTM": train_lstm,
        "GRU": train_gru,
    }

    results = {}
    best_model_name = None
    best_r2 = -1e9

    for name, func in models.items():
        print(f"\nTraining {name}...", end=" ")
        result = func(train, test)
        results[name] = result
        m = result["metrics"]
        print(f"MAE: {m['MAE']:.2f}, RMSE: {m['RMSE']:.2f}, R2: {m['R2']:.4f}, SMAPE: {m.get('SMAPE', np.nan):.2f}%")
        if not np.isnan(m["R2"]) and m["R2"] > best_r2:
            best_r2 = m["R2"]
            best_model_name = name

    summary = {}
    for name, res in results.items():
        summary[name] = res["metrics"]
    summary["best_model"] = best_model_name
    summary["test_size"] = test_size
    summary["train_end"] = str(train["ds"].max())
    summary["test_start"] = str(test["ds"].min())
    summary["test_end"] = str(test["ds"].max())

    with open(RESULTS_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"BEST MODEL: {best_model_name} (R2 = {best_r2:.4f})")
    print(f"{'=' * 60}")

    best_res = results[best_model_name]
    if best_res["model"] is not None:
        model_path = os.path.join(MODEL_DIR, "best_model")
        if best_model_name == "ARIMA":
            joblib.dump(best_res["model"], model_path + ".pkl")
        elif best_model_name == "Prophet":
            joblib.dump(best_res["model"], model_path + "_prophet.pkl")
        elif best_model_name in ["LSTM", "GRU"]:
            best_res["model"].save(model_path + ".keras")
            joblib.dump({"scaler": best_res["scaler"], "lookback": best_res.get("lookback", 30)}, model_path + "_meta.pkl")
        else:
            joblib.dump({"model": best_res["model"], "scaler": best_res.get("scaler"),
                         "feature_cols": best_res.get("feature_cols")}, model_path + ".pkl")

        meta = {"model_name": best_model_name, "metrics": best_res["metrics"], "test_size": test_size}
        with open(os.path.join(MODEL_DIR, "best_model_info.json"), "w") as f:
            json.dump(meta, f, indent=2)

    return results, best_model_name, daily, train, test


if __name__ == "__main__":
    train_all()
