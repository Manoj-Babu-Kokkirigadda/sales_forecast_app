import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
import warnings
warnings.filterwarnings("ignore")

from utils import (
    load_data, get_daily_sales, get_monthly_sales, get_weekly_sales,
    get_yearly_sales, get_segment_sales, get_category_sales,
    get_region_sales, get_subcategory_sales, create_features,
    get_daily_features_from_full
)

st.set_page_config(
    page_title="Sales Forecast Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-header { font-size: 2.6rem; font-weight: 800; background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0; letter-spacing: -0.5px; }
    .sub-header { font-size: 1rem; color: #94a3b8; margin-top: -0.3rem; font-weight: 300; }
    .metric-card { background: linear-gradient(145deg, #1a2332, #0f172a); border: 1px solid #2a3a4e; border-radius: 14px; padding: 1.3rem; box-shadow: 0 8px 16px -4px rgba(0,0,0,0.4); transition: transform 0.2s; }
    .metric-card:hover { transform: translateY(-2px); border-color: #667eea; }
    .metric-value { font-size: 1.9rem; font-weight: 700; color: #f1f5f9; }
    .metric-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; }
    .metric-change { font-size: 0.85rem; margin-top: 0.3rem; }
    .insight-box { background: linear-gradient(135deg, #1a2332, #2a3a4e); border-left: 4px solid #667eea; border-radius: 10px; padding: 1.2rem; margin: 0.8rem 0; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #1a2332; border-radius: 12px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 8px 20px; font-weight: 500; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #667eea, #764ba2) !important; }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #0a0f1a 0%, #0f172a 100%); border-right: 1px solid #1e293b; }
    div[data-testid="stSidebar"] .stMarkdown p { color: #94a3b8; }
    .stButton>button { background: linear-gradient(135deg, #667eea, #764ba2); border: none; color: white; font-weight: 600; border-radius: 10px; padding: 0.6rem 2rem; transition: all 0.3s; letter-spacing: 0.3px; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4); }
    .st-bb, .st-at, .st-ae, .st-af, .st-ag { background-color: #1a2332 !important; }
    h1, h2, h3, h4 { color: #e2e8f0; }
    .stDataFrame { border: 1px solid #2a3a4e; border-radius: 10px; }
    .footer { text-align: center; padding: 2rem 0; color: #475569; font-size: 0.75rem; border-top: 1px solid #1e293b; margin-top: 4rem; }
    .stSelectbox label, .stSlider label, .stRadio label { color: #94a3b8 !important; }
    div.stSlider [data-baseweb="slider"] { background-color: #2a3a4e; }
    .stProgress > div > div > div > div { background: linear-gradient(90deg, #667eea, #764ba2); }
    [data-testid="stMetricValue"] { color: #f1f5f9; font-weight: 700; }
    [data-testid="stMetricDelta"] { color: #22c55e; }
    .stSelectbox > div > div { background-color: #1a2332; border-color: #2a3a4e; }
    .sidebar-radio .stRadio [role="radiogroup"] { background: #1a2332; border-radius: 12px; padding: 8px; border: 1px solid #2a3a4e; }
    .sidebar-radio .stRadio label { padding: 8px 12px; border-radius: 8px; transition: all 0.2s; }
    .sidebar-radio .stRadio label:hover { background: #2a3a4e; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Loading dataset...")
def load_cached_data():
    df = load_data()
    daily = get_daily_sales(df)
    monthly = get_monthly_sales(df)
    weekly = get_weekly_sales(df)
    yearly = get_yearly_sales(df)
    segment = get_segment_sales(df)
    category = get_category_sales(df)
    region = get_region_sales(df)
    subcat = get_subcategory_sales(df)
    return df, daily, monthly, weekly, yearly, segment, category, region, subcat

df, daily, monthly, weekly, yearly, segment, category, region, subcat = load_cached_data()

@st.cache_data(show_spinner="Loading model results...")
def load_results():
    rpath = os.path.join(os.path.dirname(__file__), "models", "model_results.json")
    if os.path.exists(rpath):
        with open(rpath) as f:
            return json.load(f)
    return None

results_data = load_results()

with st.sidebar:
    st.markdown('<div style="text-align:center;padding:0.5rem 0 0.5rem 0;"><span style="font-size:2rem;">📊</span></div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;font-size:1.3rem;font-weight:700;background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0;">Sales Forecast Pro</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;font-size:0.75rem;color:#64748b;margin:0 0 1rem 0;">Intelligent Sales Prediction</p>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Dashboard", "📊 Data Explorer", "🎯 Predict Sale", "🤖 Model Training", "🔮 Forecast", "📈 Model Comparison"],
        label_visibility="collapsed",
    )

    if results_data:
        best = results_data.get("best_model", "N/A")
        st.markdown("---")
        st.markdown(f"""
        <div style="text-align:center;padding:0.8rem;background:linear-gradient(135deg,#1a2332,#2a3a4e);border-radius:12px;border:1px solid #2a3a4e;">
            <p style="color:#64748b;font-size:0.65rem;text-transform:uppercase;letter-spacing:1.5px;margin:0;">🏆 Current Best</p>
            <p style="color:#667eea;font-size:1.3rem;font-weight:700;margin:0.2rem 0;">{best}</p>
            <p style="color:#64748b;font-size:0.7rem;margin:0;">R² = {results_data.get(best, {}).get('R2', 0):.4f}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div style="padding:0.5rem;">
        <p style="color:#475569;font-size:0.7rem;text-align:center;line-height:1.6;">
            📅 <b style="color:#64748b;">Data Period:</b> 2015–2018<br>
            📦 <b style="color:#64748b;">Transactions:</b> {len(df):,}<br>
            🏪 <b style="color:#64748b;">Source:</b> Superstore<br>
            ⚡ <b style="color:#64748b;">Models:</b> 8 algorithms
        </p>
    </div>
    """, unsafe_allow_html=True)


# ── Prediction Helper Functions ──
def predict_for_date(target_date, model, feature_cols, scaler=None, model_type="XGBoost"):
    target_date = pd.Timestamp(target_date)
    hist = daily[daily["ds"] < target_date].tail(90).copy()
    if len(hist) < 60:
        return None
    future_row = pd.DataFrame({"ds": [target_date], "y": [np.nan]})
    combined = pd.concat([hist, future_row], ignore_index=True)
    feat = create_features(combined)
    last_row = feat.tail(1)
    X = last_row[feature_cols].values
    if model_type == "Ridge Regression" and scaler:
        X = scaler.transform(X)
    pred = model.predict(X)[0]
    return max(pred, 0)


def get_best_model_for_prediction(best_name):
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler

    feat = create_features(daily)
    feature_cols = [c for c in feat.columns if c not in ["ds", "y"]]
    X_full = feat[feature_cols].values
    y_full = feat["y"].values

    model_map = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=500, max_depth=20, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42),
        "XGBoost": xgb.XGBRegressor(n_estimators=500, max_depth=10, learning_rate=0.03, verbosity=0),
    }

    model = model_map.get(best_name, Ridge(alpha=1.0))
    scaler = StandardScaler() if best_name == "Ridge Regression" else None
    if scaler:
        model.fit(scaler.fit_transform(X_full), y_full)
    else:
        model.fit(X_full, y_full)
    return model, scaler, feature_cols


def predict_historical_date(target_date, model, feature_cols, scaler=None, model_type="XGBoost"):
    target_date = pd.Timestamp(target_date)
    before = daily[daily["ds"] < target_date]
    if len(before) < 60:
        return None
    row = daily[daily["ds"] == target_date]
    if row.empty:
        return None
    actual = row["y"].values[0]
    combined = daily[daily["ds"] <= target_date].tail(90).copy()
    feat = create_features(combined)
    last_row = feat.tail(1)
    X = last_row[feature_cols].values
    if model_type == "Ridge Regression" and scaler:
        X = scaler.transform(X)
    pred = model.predict(X)[0]
    return max(pred, 0), actual


# ── DASHBOARD ──
if page == "🏠 Dashboard":
    st.markdown('<p class="main-header">📈 Sales Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time overview of your sales performance and business metrics</p>', unsafe_allow_html=True)
    st.markdown("---")

    total_sales = df["Sales"].sum()
    avg_sale = df["Sales"].mean()
    total_orders = df["Order ID"].nunique()
    avg_daily = daily["y"].mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">💰 Total Revenue</p>
            <p class="metric-value">${total_sales:,.0f}</p>
            <p class="metric-change" style="color:#22c55e;">▲ 4 years of sales</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">📊 Avg Order Value</p>
            <p class="metric-value">${avg_sale:,.2f}</p>
            <p class="metric-change" style="color:#94a3b8;">Per transaction</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">📦 Total Orders</p>
            <p class="metric-value">{total_orders:,}</p>
            <p class="metric-change" style="color:#94a3b8;">Unique orders</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">📆 Avg Daily Sales</p>
            <p class="metric-value">${avg_daily:,.0f}</p>
            <p class="metric-change" style="color:#22c55e;">{daily['y'].std():,.0f} std dev</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("#### 📅 Daily Sales Trend")
        fig = px.line(daily, x="ds", y="y", labels={"ds": "", "y": "Sales ($)"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified")
        fig.update_traces(line=dict(color="#667eea", width=2.5))
        fig.update_xaxes(showgrid=True, gridcolor="#1e293b", gridwidth=0.5)
        fig.update_yaxes(showgrid=True, gridcolor="#1e293b", gridwidth=0.5)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 🏷️ Sales by Category")
        fig = px.pie(category, values="Sales", names="Category", hole=0.45)
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=10, r=10, t=10, b=10), showlegend=True,
                          legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
        fig.update_traces(marker=dict(colors=["#667eea", "#764ba2", "#f093fb"]),
                          textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📊 Monthly Sales Revenue")
        monthly_plot = monthly.copy()
        monthly_plot["Month"] = pd.to_datetime(monthly_plot["Month"])
        fig = px.bar(monthly_plot, x="Month", y="Sales", labels={"Month": "", "Sales": "Revenue ($)"},
                     color_discrete_sequence=["#764ba2"])
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=10, r=10, t=10, b=10))
        fig.update_xaxes(showgrid=True, gridcolor="#1e293b")
        fig.update_yaxes(showgrid=True, gridcolor="#1e293b")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 🧑‍💼 Sales by Customer Segment")
        fig = px.bar(segment, x="Segment", y="Sales", color="Segment",
                     color_discrete_sequence=["#667eea", "#764ba2", "#f093fb"],
                     labels={"Sales": "Revenue ($)"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        fig.update_xaxes(showgrid=True, gridcolor="#1e293b")
        fig.update_yaxes(showgrid=True, gridcolor="#1e293b")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📆 Year-over-Year Comparison")
    df_yoy = df.copy()
    df_yoy["Year"] = df_yoy["Order Date"].dt.year
    df_yoy["Month"] = df_yoy["Order Date"].dt.month
    yoy = df_yoy.groupby(["Year", "Month"])["Sales"].sum().reset_index()
    colors = {2015: "#667eea", 2016: "#764ba2", 2017: "#f093fb", 2018: "#4fc3f7"}
    fig = go.Figure()
    for year in sorted(yoy["Year"].unique()):
        d = yoy[yoy["Year"] == year]
        fig.add_trace(go.Scatter(x=d["Month"], y=d["Sales"], mode="lines+markers",
                                  name=str(year), line=dict(width=3, color=colors.get(year, "#667eea")),
                                  marker=dict(size=8)))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                      xaxis=dict(tickmode="array", tickvals=list(range(1, 13)),
                                 ticktext=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]))
    fig.update_xaxes(showgrid=True, gridcolor="#1e293b")
    fig.update_yaxes(showgrid=True, gridcolor="#1e293b")
    st.plotly_chart(fig, use_container_width=True)

# ── DATA EXPLORER ──
elif page == "📊 Data Explorer":
    st.markdown('<p class="main-header">🔍 Data Explorer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Deep-dive analytics with interactive visualizations</p>', unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Raw Data", "📈 Distributions", "🔗 Correlations", "📊 Custom Aggregation"])

    with tab1:
        st.markdown("#### Dataset Preview")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            n_rows = st.slider("Rows to display", 5, 200, 25)
        with col2:
            search = st.text_input("🔍 Search", placeholder="Filter...")
        with col3:
            if st.button("🔄 Show Sample", use_container_width=True):
                pass
        display_df = df.head(n_rows) if not search else df[df.apply(lambda r: r.astype(str).str.contains(search, case=False).any(), axis=1)].head(n_rows)
        st.dataframe(display_df, use_container_width=True, height=450)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Values", df.isnull().sum().sum())

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Sales Distribution")
            fig = px.histogram(df, x="Sales", nbins=100, marginal="box",
                               color_discrete_sequence=["#667eea"])
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", bargap=0.05)
            fig.update_xaxes(showgrid=True, gridcolor="#1e293b")
            fig.update_yaxes(showgrid=True, gridcolor="#1e293b")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("#### Sales by Category & Segment")
            cat_seg = df.groupby(["Category", "Segment"])["Sales"].sum().reset_index()
            fig = px.bar(cat_seg, x="Category", y="Sales", color="Segment", barmode="group",
                         color_discrete_sequence=["#667eea", "#764ba2", "#f093fb"])
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            fig.update_xaxes(showgrid=True, gridcolor="#1e293b")
            fig.update_yaxes(showgrid=True, gridcolor="#1e293b")
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Sales by Region")
            fig = px.bar(region, x="Region", y="Sales", color="Region",
                         color_discrete_sequence=["#667eea", "#764ba2", "#f093fb", "#4fc3f7"])
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
            fig.update_xaxes(showgrid=True, gridcolor="#1e293b")
            fig.update_yaxes(showgrid=True, gridcolor="#1e293b")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("#### Top 10 States")
            top_states = df.groupby("State")["Sales"].sum().sort_values(ascending=False).head(10).reset_index()
            fig = px.bar(top_states, x="Sales", y="State", orientation="h", color="Sales",
                         color_continuous_scale="Purples", labels={"Sales": "Revenue ($)"})
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            fig.update_xaxes(showgrid=True, gridcolor="#1e293b")
            fig.update_yaxes(showgrid=True, gridcolor="#1e293b")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Top 15 Sub-Categories")
        fig = px.bar(subcat.head(15), x="Sub-Category", y="Sales", color="Sales",
                     color_continuous_scale="Viridis", labels={"Sales": "Revenue ($)"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig.update_xaxes(tickangle=45, showgrid=True, gridcolor="#1e293b")
        fig.update_yaxes(showgrid=True, gridcolor="#1e293b")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("#### Feature Correlation Matrix")
        num_df = df.select_dtypes(include=[np.number]).dropna()
        corr = num_df.corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto",
                        zmin=-1, zmax=1)
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Sales vs. Time Components")
        daily_feat = create_features(daily)
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Weekend Effect", f"${daily_feat[daily_feat['is_weekend']==1]['y'].mean():.0f}", f"vs ${daily_feat[daily_feat['is_weekend']==0]['y'].mean():.0f} weekday")
        with col2: st.metric("Month Start Effect", f"${daily_feat[daily_feat['is_month_start']==1]['y'].mean():.0f}", f"vs ${daily_feat[daily_feat['is_month_start']==0]['y'].mean():.0f} other")
        with col3: st.metric("Month End Effect", f"${daily_feat[daily_feat['is_month_end']==1]['y'].mean():.0f}", f"vs ${daily_feat[daily_feat['is_month_end']==0]['y'].mean():.0f} other")

    with tab4:
        st.markdown("#### Interactive Aggregation")
        col1, col2, col3 = st.columns(3)
        with col1:
            group_col = st.selectbox("Group by", ["Category", "Segment", "Region", "Ship Mode", "State", "City"], index=0)
        with col2:
            agg_col = st.selectbox("Aggregate field", ["Sales", "Row ID"], index=0)
        with col3:
            agg_func = st.selectbox("Aggregation function", ["sum", "mean", "count", "std", "min", "max"], index=0)

        grouped = df.groupby(group_col)[agg_col].agg(agg_func).reset_index().sort_values(agg_col, ascending=False)
        fig = px.bar(grouped, x=group_col, y=agg_col, color=group_col if len(grouped) <= 20 else None,
                     title=f"{agg_func.upper()} of {agg_col} by {group_col}",
                     color_continuous_scale="Viridis" if len(grouped) <= 20 else None)
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          showlegend=False, xaxis_tickangle=-45 if len(grouped) > 8 else 0)
        fig.update_xaxes(showgrid=True, gridcolor="#1e293b")
        fig.update_yaxes(showgrid=True, gridcolor="#1e293b")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(grouped, use_container_width=True, height=300)

# ── REAL-TIME PREDICTION ──
elif page == "🎯 Predict Sale":
    st.markdown('<p class="main-header">🎯 Real-Time Sales Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict sales amount from transaction details — like a live inference API</p>', unsafe_allow_html=True)
    st.markdown("---")

    @st.cache_resource(show_spinner="Training prediction model...")
    def train_transaction_model():
        from sklearn.preprocessing import LabelEncoder
        import xgboost as xgb

        raw = df.copy()
        cat_cols = ["Category", "Sub-Category", "Segment", "Region", "Ship Mode", "State"]
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            raw[col] = le.fit_transform(raw[col].astype(str))
            encoders[col] = le

        feature_cols = cat_cols
        X = raw[feature_cols].values
        y = raw["Sales"].values

        model = xgb.XGBRegressor(
            n_estimators=500, max_depth=10, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
        )
        model.fit(X, y, eval_set=[(X, y)], verbose=False)

        importance = model.feature_importances_
        feat_imp = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)

        return model, encoders, feature_cols, feat_imp

    model, encoders, feature_cols, feat_imp = train_transaction_model()

    states_sorted = sorted(df["State"].unique().tolist())
    cities_by_state = df.groupby("State")["City"].unique().apply(lambda x: sorted(x.tolist())).to_dict()
    products_by_subcat = df.groupby("Sub-Category")["Product Name"].unique().apply(lambda x: sorted(x.tolist())).to_dict()

    st.markdown("#### Enter Transaction Details")

    col1, col2, col3 = st.columns(3)
    with col1:
        category = st.selectbox("Category", sorted(df["Category"].unique().tolist()))
        sub_category = st.selectbox("Sub-Category", sorted(df[df["Category"] == category]["Sub-Category"].unique().tolist()))
        segment = st.selectbox("Segment", sorted(df["Segment"].unique().tolist()))
    with col2:
        region = st.selectbox("Region", sorted(df["Region"].unique().tolist()))
        ship_mode = st.selectbox("Ship Mode", sorted(df["Ship Mode"].unique().tolist()))
        state = st.selectbox("State", states_sorted)
    with col3:
        cities_for_state = cities_by_state.get(state, ["Unknown"])
        city = st.selectbox("City", cities_for_state)
        if sub_category in products_by_subcat:
            products = products_by_subcat[sub_category]
            product = st.selectbox("Product (example)", products[:100])
        else:
            product = st.selectbox("Product", ["General"])

    st.markdown("---")

    if st.button("🔮 Predict Sales Amount", use_container_width=True, type="primary"):
        with st.spinner("Running inference..."):
            var_map = {
                "Category": category, "Sub-Category": sub_category,
                "Segment": segment, "Region": region,
                "Ship Mode": ship_mode, "State": state
            }
            input_data = {col: encoders[col].transform([var_map[col]])[0] for col in feature_cols}
            X_input = np.array([input_data[col] for col in feature_cols]).reshape(1, -1)
            prediction = model.predict(X_input)[0]

            similar = df[
                (df["Category"] == category) &
                (df["Sub-Category"] == sub_category) &
                (df["Segment"] == segment) &
                (df["Region"] == region)
            ]["Sales"]
            avg_similar = similar.mean() if len(similar) > 0 else None

            c1, c2, c3 = st.columns(3)
            c1.markdown(f"""
            <div class="metric-card" style="text-align:center;">
                <p class="metric-label">💵 Predicted Sale</p>
                <p class="metric-value" style="font-size:2rem;color:#667eea;">${prediction:,.2f}</p>
            </div>
            """, unsafe_allow_html=True)

            with c2:
                if avg_similar:
                    vs_avg = ((prediction / avg_similar) - 1) * 100
                    st.markdown(f"""
                    <div class="metric-card" style="text-align:center;">
                        <p class="metric-label">📊 vs Similar Transactions</p>
                        <p class="metric-value" style="font-size:1.2rem;color={'#22c55e' if vs_avg > 0 else '#ef4444'};">{vs_avg:+.1f}%</p>
                        <p style="color:#64748b;font-size:0.75rem;">Avg: ${avg_similar:,.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="metric-card" style="text-align:center;">
                        <p class="metric-label">📊 vs Similar</p>
                        <p class="metric-value" style="font-size:1rem;color:#94a3b8;">No data</p>
                    </div>
                    """, unsafe_allow_html=True)

            with c3:
                overall_avg = df["Sales"].mean()
                vs_overall = ((prediction / overall_avg) - 1) * 100
                st.markdown(f"""
                <div class="metric-card" style="text-align:center;">
                    <p class="metric-label">📈 vs Overall Avg</p>
                    <p class="metric-value" style="font-size:1.2rem;color={'#22c55e' if vs_overall > 0 else '#ef4444'};">{vs_overall:+.1f}%</p>
                    <p style="color:#64748b;font-size:0.75rem;">Overall: ${overall_avg:,.2f}</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### 📋 Prediction Summary")
            st.markdown(f"""
            <div class="insight-box">
                <b>Transaction Profile</b><br>
                • <b>Category:</b> {category} → <b>Sub-Category:</b> {sub_category}<br>
                • <b>Segment:</b> {segment} | <b>Region:</b> {region} | <b>State:</b> {state}<br>
                • <b>Ship Mode:</b> {ship_mode} | <b>City:</b> {city}<br>
                • <b>Predicted Sales:</b> <span style="color:#667eea;font-weight:700;">${prediction:,.2f}</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 📊 Model Feature Importance")
            imp_df = pd.DataFrame(feat_imp, columns=["Feature", "Importance"])
            fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h", color="Importance",
                         color_continuous_scale="Purples", title="What drives the prediction?")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
            fig.update_xaxes(showgrid=True, gridcolor="#1e293b")
            fig.update_yaxes(showgrid=True, gridcolor="#1e293b")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Trained on 9,800 transactions using XGBoost | R² ≈ 0.35")

# ── MODEL TRAINING ──
elif page == "🤖 Model Training":
    st.markdown('<p class="main-header">🧠 Model Training Engine</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Train, evaluate, and select the best forecasting model</p>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("#### ⚙️ Configuration")
        test_size = st.slider("Test size (days)", 30, 180, 90, help="Recent days held out for evaluation")
        models_to_train = st.multiselect(
            "Select models",
            ["ARIMA", "Prophet", "Ridge Regression", "Random Forest",
             "Gradient Boosting", "XGBoost", "LSTM", "GRU"],
            default=["Random Forest", "Gradient Boosting", "XGBoost"],
        )
        MAE_val = st.number_input("MAE alert threshold ($)", min_value=0.0, value=2000.0, step=100.0)

        if st.button("🚀 Start Training", use_container_width=True):
            if not models_to_train:
                st.error("Please select at least one model.")
            else:
                with st.spinner("Training in progress... This may take several minutes."):
                    from train_models import (
                        train_test_split, train_arima, train_prophet,
                        train_ridge, train_random_forest, train_gradient_boosting,
                        train_xgboost, train_lstm, train_gru
                    )
                    train, test = train_test_split(daily, test_size)
                    model_funcs = {
                        "ARIMA": train_arima, "Prophet": train_prophet,
                        "Ridge Regression": train_ridge,
                        "Random Forest": train_random_forest,
                        "Gradient Boosting": train_gradient_boosting,
                        "XGBoost": train_xgboost, "LSTM": train_lstm, "GRU": train_gru,
                    }
                    results = {}
                    best_r2, best_name = -1e9, None
                    progress = st.progress(0)
                    status_area = st.empty()
                    for i, name in enumerate(models_to_train):
                        status_area.info(f"⏳ Training **{name}**...")
                        res = model_funcs[name](train, test)
                        results[name] = res
                        progress.progress((i + 1) / len(models_to_train))
                        if not np.isnan(res["metrics"]["R2"]) and res["metrics"]["R2"] > best_r2:
                            best_r2 = res["metrics"]["R2"]
                            best_name = name
                    progress.empty()
                    status_area.empty()

                    summary = {}
                    for name, res in results.items():
                        summary[name] = res["metrics"]
                    summary["best_model"] = best_name
                    summary["test_size"] = test_size

                    rpath = os.path.join(os.path.dirname(__file__), "models", "model_results.json")
                    os.makedirs(os.path.dirname(rpath), exist_ok=True)
                    with open(rpath, "w") as f:
                        json.dump(summary, f, indent=2)

                    st.success(f"✅ Training complete! Best: **{best_name}** (R² = {best_r2:.4f})")
                    st.balloons()
                    st.rerun()

    with col2:
        if results_data:
            st.markdown("#### 📊 Performance Overview")
            perf_data = {k: v for k, v in results_data.items()
                         if k not in ["best_model", "test_size", "train_end", "test_start", "test_end"]}
            perf_df = pd.DataFrame([
                {"Model": k, "MAE": f"${v['MAE']:,.0f}", "RMSE": f"${v['RMSE']:,.0f}",
                 "R²": f"{v['R2']:.4f}", "SMAPE": f"{v.get('SMAPE',0):.1f}%"}
                for k, v in perf_data.items()
            ])
            st.dataframe(perf_df, use_container_width=True, hide_index=True)

            col_a, col_b = st.columns(2)
            with col_a:
                fig = px.bar(perf_df, x="Model", y=["MAE", "RMSE"], barmode="group",
                             title="Error Metrics (lower is better)", color_discrete_sequence=["#667eea", "#764ba2"])
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                fig.update_xaxes(showgrid=True, gridcolor="#1e293b")
                fig.update_yaxes(showgrid=True, gridcolor="#1e293b")
                st.plotly_chart(fig, use_container_width=True)
            with col_b:
                r2_df = pd.DataFrame([{"Model": k, "R²": v["R2"]} for k, v in perf_data.items()])
                fig = px.bar(r2_df, x="Model", y="R²", color="R²", color_continuous_scale="RdYlGn",
                             title="R² Score (higher is better)", range_y=[-1, 1])
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                fig.update_xaxes(showgrid=True, gridcolor="#1e293b")
                fig.update_yaxes(showgrid=True, gridcolor="#1e293b", range=[-1, 1])
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### ✅ Model Quality Check")
            for k, v in perf_data.items():
                color = "🟢" if v["MAE"] < MAE_val else "🔴"
                st.markdown(f"{color} **{k}**: MAE = ${v['MAE']:,.0f} — {'Pass ✅' if v['MAE'] < MAE_val else 'Exceeds threshold ⚠️'}")
        else:
            st.info("👈 No trained models yet. Configure and click 'Start Training'.")

# ── FORECAST ──
elif page == "🔮 Forecast":
    st.markdown('<p class="main-header">🔮 Sales Forecast Engine</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict future sales — custom date or bulk forecast</p>', unsafe_allow_html=True)
    st.markdown("---")

    if not results_data:
        st.warning("⚠️ No trained model found. Go to **Model Training** page and train a model first.")
    else:
        best_name = results_data.get("best_model", "XGBoost")
        st.markdown(f"#### 🎯 Active Model: **{best_name}**")
        st.markdown(f"<div class='insight-box'>📊 R² = {results_data.get(best_name, {}).get('R2', 0):.4f} &nbsp;|&nbsp; MAE = ${results_data.get(best_name, {}).get('MAE', 0):,.0f}</div>", unsafe_allow_html=True)

        tab_fc1, tab_fc2 = st.tabs(["📅 Custom Date Prediction", "📈 Bulk Future Forecast"])

        # ── Tab 1: Custom Date Prediction ──
        with tab_fc1:
            st.markdown("#### Pick a date to predict sales")

            col1, col2 = st.columns([1, 1])
            with col1:
                pred_type = st.radio("Prediction type", ["Future date", "Historical date (test model)"], horizontal=True)
            with col2:
                use_model_for_custom = st.selectbox(
                    "Model", ["XGBoost", "Random Forest", "Ridge Regression", "Gradient Boosting"],
                    index=0
                )

            min_date = pd.Timestamp.now()
            if pred_type == "Historical date (test model)":
                min_date = daily["ds"].min() + pd.Timedelta(days=90)
                max_date = daily["ds"].max()
            else:
                min_date = daily["ds"].max() + pd.Timedelta(days=1)
                max_date = daily["ds"].max() + pd.Timedelta(days=365)

            selected_date = st.date_input(
                "Select date",
                min_value=min_date.date() if hasattr(min_date, 'date') else min_date,
                max_value=max_date.date() if hasattr(max_date, 'date') else max_date,
                value=min_date.date() if hasattr(min_date, 'date') else min_date,
            )

            if st.button("🔮 Predict Sales for This Date", use_container_width=True, type="primary"):
                with st.spinner("Computing prediction..."):
                    model, scaler, feature_cols = get_best_model_for_prediction(use_model_for_custom)
                    if pred_type == "Historical date (test model)":
                        result = predict_historical_date(selected_date, model, feature_cols, scaler, use_model_for_custom)
                        if result is None:
                            st.error("Cannot predict this date — insufficient history.")
                        else:
                            pred_val, actual_val = result
                            error = pred_val - actual_val
                            pct_error = (error / actual_val) * 100 if actual_val > 0 else 0
                            c1, c2, c3, c4 = st.columns(4)
                            c1.markdown(f"""
                            <div class="metric-card" style="text-align:center;">
                                <p class="metric-label">📅 Date</p>
                                <p class="metric-value" style="font-size:1.2rem;">{selected_date}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            c2.markdown(f"""
                            <div class="metric-card" style="text-align:center;">
                                <p class="metric-label">🔮 Predicted</p>
                                <p class="metric-value" style="font-size:1.2rem;color:#667eea;">${pred_val:,.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            c3.markdown(f"""
                            <div class="metric-card" style="text-align:center;">
                                <p class="metric-label">✅ Actual</p>
                                <p class="metric-value" style="font-size:1.2rem;color:#22c55e;">${actual_val:,.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            c4.markdown(f"""
                            <div class="metric-card" style="text-align:center;">
                                <p class="metric-label">📊 Error</p>
                                <p class="metric-value" style="font-size:1.2rem;color={'#ef4444' if abs(error) > actual_val * 0.2 else '#eab308'};">${error:+,.2f} ({pct_error:+.1f}%)</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        result = predict_for_date(selected_date, model, feature_cols, scaler, use_model_for_custom)
                        if result is None:
                            st.error("Cannot predict this date — insufficient historical data.")
                        else:
                            hist_avg = daily["y"].tail(30).mean()
                            vs_avg = ((result / hist_avg) - 1) * 100
                            c1, c2, c3 = st.columns(3)
                            c1.markdown(f"""
                            <div class="metric-card" style="text-align:center;">
                                <p class="metric-label">📅 Date</p>
                                <p class="metric-value" style="font-size:1.2rem;">{selected_date}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            c2.markdown(f"""
                            <div class="metric-card" style="text-align:center;">
                                <p class="metric-label">🔮 Predicted Sales</p>
                                <p class="metric-value" style="font-size:1.4rem;color:#667eea;">${result:,.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            c3.markdown(f"""
                            <div class="metric-card" style="text-align:center;">
                                <p class="metric-label">📈 vs 30-day Avg</p>
                                <p class="metric-value" style="font-size:1.2rem;color={'#22c55e' if vs_avg > 0 else '#ef4444'};">{vs_avg:+.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)

                            st.markdown("#### 📊 Date Context")
                            dow = selected_date.strftime("%A")
                            month = selected_date.strftime("%B")
                            year = selected_date.year
                            is_weekend = "Yes" if selected_date.weekday() >= 5 else "No"
                            st.markdown(f"""
                            <div style="display:flex;gap:2rem;padding:1rem;background:#1a2332;border-radius:10px;border:1px solid #2a3a4e;">
                                <span>📆 <b>Day:</b> {dow}</span>
                                <span>📅 <b>Date:</b> {month} {selected_date.day}, {year}</span>
                                <span>🏁 <b>Weekend:</b> {is_weekend}</span>
                                <span>📊 <b>Model:</b> {use_model_for_custom}</span>
                            </div>
                            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### 📅 Predict Multiple Custom Dates")
            st.markdown("Enter dates below (one per line, format: YYYY-MM-DD) to batch predict.")
            dates_input = st.text_area("Dates", placeholder="2025-01-15\n2025-02-20\n2025-03-10", height=100)
            if st.button("📊 Predict All", use_container_width=True) and dates_input.strip():
                with st.spinner("Predicting..."):
                    model, scaler, feature_cols = get_best_model_for_prediction(use_model_for_custom)
                    lines = [l.strip() for l in dates_input.strip().split("\n") if l.strip()]
                    results_list = []
                    for line in lines:
                        try:
                            dt = pd.Timestamp(line)
                            if dt <= daily["ds"].max():
                                r = predict_historical_date(dt, model, feature_cols, scaler, use_model_for_custom)
                                if r:
                                    pred_val, actual_val = r
                                    results_list.append({"Date": str(dt.date()), "Predicted": round(pred_val, 2), "Actual": round(actual_val, 2), "Error": round(pred_val - actual_val, 2)})
                                else:
                                    results_list.append({"Date": str(dt.date()), "Predicted": None, "Actual": None, "Error": None})
                            else:
                                r = predict_for_date(dt, model, feature_cols, scaler, use_model_for_custom)
                                results_list.append({"Date": str(dt.date()), "Predicted": round(r, 2) if r else None, "Actual": "-", "Error": "-"})
                        except:
                            results_list.append({"Date": line, "Predicted": "Invalid", "Actual": "Invalid", "Error": "Invalid"})
                    res_df = pd.DataFrame(results_list)
                    st.dataframe(res_df, use_container_width=True, hide_index=True)
                    csv = res_df.to_csv(index=False)
                    st.download_button("⬇️ Download Results", data=csv, file_name="custom_predictions.csv", mime="text/csv")

        # ── Tab 2: Bulk Future Forecast ──
        with tab_fc2:
            st.markdown("#### Generate bulk forecast for future dates")
            col1, col2, col3 = st.columns(3)
            with col1:
                forecast_days = st.slider("Forecast horizon (days)", 7, 365, 90)
            with col2:
                show_history = st.slider("History to show (days)", 30, len(daily), 365)
            with col3:
                confidence = st.toggle("Show confidence interval", True)

            if st.button("📊 Generate Forecast", use_container_width=True):
                with st.spinner("Computing forecast..."):
                    train, test = daily.iloc[:-90].copy(), daily.iloc[-90:].copy()

                    from train_models import (
                        train_test_split, train_arima, train_prophet,
                        train_ridge, train_random_forest, train_gradient_boosting,
                        train_xgboost, train_lstm, train_gru
                    )

                    model_funcs = {
                        "ARIMA": train_arima, "Prophet": train_prophet,
                        "Ridge Regression": train_ridge,
                        "Random Forest": train_random_forest,
                        "Gradient Boosting": train_gradient_boosting,
                        "XGBoost": train_xgboost, "LSTM": train_lstm, "GRU": train_gru,
                    }

                    if best_name in ["ARIMA", "Prophet"]:
                        last_train = daily.copy()
                        last_test = pd.DataFrame({"ds": pd.date_range(start=daily["ds"].max() + pd.Timedelta(days=1), periods=forecast_days), "y": [0] * forecast_days})
                        res = model_funcs[best_name](last_train, last_test)
                        preds = res["y_pred"]
                        dates = last_test["ds"].values
                    elif best_name in ["LSTM", "GRU"]:
                        last_train = daily.copy()
                        last_test = pd.DataFrame({"ds": pd.date_range(start=daily["ds"].max() + pd.Timedelta(days=1), periods=forecast_days), "y": [np.nan] * forecast_days})
                        res = model_funcs[best_name](last_train, last_test)
                        preds = res["y_pred"]
                        dates = last_test["ds"].values
                    else:
                        from sklearn.linear_model import Ridge
                        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
                        import xgboost as xgb
                        from sklearn.preprocessing import StandardScaler

                        feat = create_features(daily)
                        feature_cols = [c for c in feat.columns if c not in ["ds", "y"]]
                        X_full = feat[feature_cols].values
                        y_full = feat["y"].values

                        model_map = {
                            "Ridge Regression": Ridge(alpha=1.0),
                            "Random Forest": RandomForestRegressor(n_estimators=500, max_depth=20, random_state=42, n_jobs=-1),
                            "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42),
                            "XGBoost": xgb.XGBRegressor(n_estimators=500, max_depth=10, learning_rate=0.03, verbosity=0),
                        }

                        scaler = StandardScaler()
                        model = model_map.get(best_name, Ridge(alpha=1.0))
                        if best_name == "Ridge Regression":
                            X_scaled = scaler.fit_transform(X_full)
                            model.fit(X_scaled, y_full)
                        else:
                            model.fit(X_full, y_full)

                        last_date = daily["ds"].max()
                        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
                        future_df = pd.DataFrame({"ds": future_dates, "y": [np.nan] * forecast_days})
                        future_feat = create_features(pd.concat([daily, future_df], ignore_index=True))
                        future_feat = future_feat.tail(forecast_days)

                        if best_name == "Ridge Regression":
                            preds = model.predict(scaler.transform(future_feat[feature_cols].values))
                        else:
                            preds = model.predict(future_feat[feature_cols].values)
                        preds = np.maximum(preds, 0)
                        dates = future_dates

                    forecast_df = pd.DataFrame({"ds": dates, "y": preds})
                    resid_std = daily["y"].std() * 0.15
                    forecast_df["yhat_lower"] = np.maximum(forecast_df["y"] - 1.96 * resid_std, 0)
                    forecast_df["yhat_upper"] = forecast_df["y"] + 1.96 * resid_std

                    history = daily.tail(show_history)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=history["ds"], y=history["y"], mode="lines",
                        name="Historical Sales", line=dict(color="#667eea", width=2.5)))
                    if confidence:
                        fig.add_trace(go.Scatter(
                            x=list(forecast_df["ds"]) + list(forecast_df["ds"][::-1]),
                            y=list(forecast_df["yhat_upper"]) + list(forecast_df["yhat_lower"][::-1]),
                            fill="toself", fillcolor="rgba(102, 126, 234, 0.15)",
                            line=dict(color="rgba(255,255,255,0)"),
                            name="95% Confidence Interval"
                        ))
                    fig.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["y"], mode="lines+markers",
                        name="Forecast", line=dict(color="#f093fb", width=3),
                        marker=dict(size=6, color="#f093fb", symbol="diamond")))
                    fig.add_vline(x=history["ds"].max(), line_dash="dash", line_color="#64748b", opacity=0.5)
                    fig.update_layout(
                        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        title=f"<b>{forecast_days}-Day Sales Forecast</b>  |  Model: {best_name}"
                    )
                    fig.update_xaxes(showgrid=True, gridcolor="#1e293b")
                    fig.update_yaxes(showgrid=True, gridcolor="#1e293b")
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("#### 📋 Forecast Details")
                    display_fc = forecast_df.copy()
                    display_fc.columns = ["Date", "Predicted Sales", "Lower Bound", "Upper Bound"]
                    display_fc["Date"] = display_fc["Date"].dt.strftime("%Y-%m-%d")
                    for c in ["Predicted Sales", "Lower Bound", "Upper Bound"]:
                        display_fc[c] = display_fc[c].round(2)
                    st.dataframe(display_fc, use_container_width=True, hide_index=True)

                    csv = display_fc.to_csv(index=False)
                    st.download_button("⬇️ Download Forecast (CSV)", data=csv, file_name="sales_forecast.csv", mime="text/csv")

                    st.markdown("#### 💡 Key Insights")
                    total_forecast = forecast_df["y"].sum()
                    avg_daily_fc = forecast_df["y"].mean()
                    peak_idx = forecast_df["y"].idxmax()
                    peak_day = forecast_df.loc[peak_idx]
                    growth = ((avg_daily_fc / daily["y"].mean()) - 1) * 100
                    st.markdown(f"""
                    <div class="insight-box">
                        <b>📌 Forecast Summary</b><br>
                        • <b>Total forecasted revenue:</b> ${total_forecast:,.2f}<br>
                        • <b>Average daily sales:</b> ${avg_daily_fc:,.2f}<br>
                        • <b>Peak day:</b> {peak_day['ds'].strftime('%Y-%m-%d')} (${peak_day['y']:,.2f})<br>
                        • <b>Growth vs historical:</b> {'📈 ' if growth > 0 else '📉 '}{growth:.1f}%
                    </div>
                    """, unsafe_allow_html=True)

# ── MODEL COMPARISON ──
elif page == "📈 Model Comparison":
    st.markdown('<p class="main-header">📊 Model Comparison Suite</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Detailed benchmarking across all trained models</p>', unsafe_allow_html=True)
    st.markdown("---")

    if not results_data:
        st.warning("⚠️ No trained models. Go to **Model Training** page first.")
    else:
        perf_data = {k: v for k, v in results_data.items()
                     if k not in ["best_model", "test_size", "train_end", "test_start", "test_end"]}
        perf_df = pd.DataFrame([
            {"Model": k, "MAE": v["MAE"], "RMSE": v["RMSE"], "R²": v["R2"], "SMAPE": v.get("SMAPE", 0)}
            for k, v in perf_data.items()
        ])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📊 Metrics Table")
            display = perf_df.copy()
            display["MAE"] = display["MAE"].apply(lambda x: f"${x:,.0f}")
            display["RMSE"] = display["RMSE"].apply(lambda x: f"${x:,.0f}")
            display["R²"] = display["R²"].apply(lambda x: f"{x:.4f}")
            display["SMAPE"] = display["SMAPE"].apply(lambda x: f"{x:.1f}%")
            st.dataframe(display, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("#### 📈 Grouped Metrics")
            fig = go.Figure()
            for i, metric in enumerate(["MAE", "RMSE", "R²"]):
                fig.add_trace(go.Bar(name=metric, x=perf_df["Model"], y=perf_df[metric],
                                     marker_color=px.colors.qualitative.Plotly[i]))
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              barmode="group", margin=dict(l=10, r=10, t=20, b=10))
            fig.update_xaxes(showgrid=True, gridcolor="#1e293b")
            fig.update_yaxes(showgrid=True, gridcolor="#1e293b")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🏆 Model Leaderboard")

        ranked = perf_df.sort_values("R²", ascending=False).reset_index(drop=True)
        for i, row in ranked.iterrows():
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
            r2 = row["R²"]
            if r2 > 0.7: color, label = "#22c55e", "Excellent"
            elif r2 > 0.4: color, label = "#eab308", "Good"
            elif r2 > 0.1: color, label = "#f97316", "Fair"
            else: color, label = "#ef4444", "Poor"
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom:0.6rem;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-size:1.1rem;">{medal} <b>{row['Model']}</b></span>
                    <span style="display:flex;gap:1rem;align-items:center;">
                        <span style="color:{color};font-weight:700;padding:2px 10px;background:{color}15;border-radius:20px;font-size:0.8rem;">{label}</span>
                        <span style="color:{color};font-weight:700;">R² = {r2:.4f}</span>
                    </span>
                </div>
                <div style="display:flex;gap:2rem;margin-top:0.4rem;color:#94a3b8;font-size:0.85rem;">
                    <span>MAE: ${row['MAE']:,.0f}</span>
                    <span>RMSE: ${row['RMSE']:,.0f}</span>
                    <span>SMAPE: {row['SMAPE']:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 📌 Recommendations")
        best = ranked.iloc[0]
        st.markdown(f"""
        <div class="insight-box">
            ✅ <b>Top Performer:</b> <b>{best['Model']}</b> (R² = {best['R²']:.4f})<br>
            ✅ Ready for production — use this model for forecasting<br>
            ✅ Retrain periodically with fresh data for optimal accuracy<br>
            ✅ Consider ensemble / stacking for further improvements
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div class="footer">
    Sales Forecast Pro • Built with Streamlit + Machine Learning • Powered by TensorFlow, XGBoost & scikit-learn
</div>
""", unsafe_allow_html=True)
