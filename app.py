import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    VotingRegressor, StackingRegressor, AdaBoostRegressor
)
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
import xgboost as xgb
import lightgbm as lgb

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Real-Time Demand Forecasting Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .main-header h1 { font-size: 2.2rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
    .main-header p  { font-size: 1rem; opacity: 0.85; margin: 0.5rem 0 0; }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        border-left: 4px solid #3498db;
        margin-bottom: 1rem;
    }
    .metric-card h3 { font-size: 0.8rem; color: #6b7280; text-transform: uppercase; letter-spacing: 1px; margin: 0; }
    .metric-card p  { font-size: 1.8rem; font-weight: 700; color: #1f2937; margin: 0.3rem 0 0; }

    .section-header {
        font-size: 1.3rem; font-weight: 700; color: #1f2937;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.4rem; margin: 1.5rem 0 1rem;
    }

    .algo-card {
        background: linear-gradient(135deg, #f8f9ff 0%, #f0f4ff 100%);
        border-radius: 12px; padding: 1rem 1.5rem;
        border: 1px solid #e0e7ff; margin-bottom: 0.8rem;
    }

    .stSelectbox > div > div { border-radius: 8px; }
    .stButton > button {
        background: linear-gradient(135deg, #2c3e50, #3498db);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; padding: 0.5rem 1.5rem;
    }
    .stButton > button:hover { opacity: 0.9; }

    .sidebar .sidebar-content { background: #1a252f; }
    div[data-testid="stSidebarContent"] { background: #1a252f; }
    div[data-testid="stSidebarContent"] * { color: white !important; }
    div[data-testid="stSidebarContent"] .stRadio > label { color: white !important; }

    .pred-result {
        background: linear-gradient(135deg, #2c3e50, #3498db);
        color: white; border-radius: 16px; padding: 2rem;
        text-align: center; margin: 1rem 0;
    }
    .pred-result h2 { font-size: 3rem; margin: 0; }
    .pred-result p  { font-size: 1rem; opacity: 0.85; margin: 0.3rem 0 0; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# DATA LOADING & FEATURE ENGINEERING
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("retail_store_inventory.csv")
    df["Date"] = pd.to_datetime(
    df["Date"],
    format="%d-%m-%Y"
)
    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Quarter"] = df["Date"].dt.quarter
    df["Year"] = df["Date"].dt.year
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["PriceDiscounted"] = df["Price"] * (1 - df["Discount"] / 100)
    df["PriceDiff"] = df["Price"] - df["Competitor Pricing"]
    df["StockUtilization"] = df["Units Sold"] / (df["Inventory Level"] + 1)
    return df

@st.cache_data
def encode_features(df):
    cat_cols = ["Category", "Region", "Seasonality", "Weather Condition"]
    le = LabelEncoder()
    df_enc = df.copy()
    for c in cat_cols:
        df_enc[c + "_enc"] = le.fit_transform(df_enc[c])
    return df_enc

@st.cache_data
def prepare_model_data(df_enc):
    feature_cols = [
        "Inventory Level", "Units Ordered", "Price", "Discount",
        "Holiday/Promotion", "Competitor Pricing", "Month", "DayOfWeek",
        "Quarter", "WeekOfYear", "PriceDiscounted", "PriceDiff",
        "StockUtilization", "Category_enc", "Region_enc",
        "Seasonality_enc", "Weather Condition_enc"
    ]
    X = df_enc[feature_cols].fillna(0)
    y = df_enc["Units Sold"]
    return X, y, feature_cols

df = load_data()
df_enc = encode_features(df)
X, y, feature_cols = prepare_model_data(df_enc)

# ──────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## System Navigation")
    page = st.radio(
        "",
        ["Dashboard Overview", "Consumer Clustering Insights",
         "Demand Forecasting Models", "Real-Time Prediction Engine"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Dataset Metadata**")
    st.markdown(f"**Records:** {len(df):,}")
    st.markdown(f"**Date Range:** {df['Date'].min().date()} to {df['Date'].max().date()}")
    st.markdown(f"**Active Stores:** {df['Store ID'].nunique()}")
    st.markdown(f"**Product Catalog:** {df['Product ID'].nunique()}")

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>Real-Time Demand Forecasting & Analytics</h1>
  <p>Consumer Behaviour Clustering & Retail Intelligence Dashboard</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════
if page == "Dashboard Overview":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <h3>Total Units Sold</h3><p>{df['Units Sold'].sum():,.0f}</p></div>""",
            unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <h3>Avg Daily Demand</h3><p>{df['Units Sold'].mean():.1f}</p></div>""",
            unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <h3>Avg Inventory Level</h3><p>{df['Inventory Level'].mean():.0f}</p></div>""",
            unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <h3>Avg Price</h3><p>${df['Price'].mean():.2f}</p></div>""",
            unsafe_allow_html=True)

    st.markdown('<div class="section-header">Monthly Sales Trend</div>', unsafe_allow_html=True)
    monthly = df.groupby(df["Date"].dt.to_period("M"))["Units Sold"].sum().reset_index()
    monthly["Date"] = monthly["Date"].astype(str)
    fig = px.area(monthly, x="Date", y="Units Sold",
                  color_discrete_sequence=["#3498db"],
                  template="plotly_white")
    fig.update_layout(xaxis_title="Month", yaxis_title="Total Units Sold",
                      hovermode="x unified")
    fig.update_traces(fill='tozeroy', line_width=2)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Sales Volume by Category</div>', unsafe_allow_html=True)
        cat_sales = df.groupby("Category")["Units Sold"].sum().reset_index()
        fig2 = px.bar(cat_sales.sort_values("Units Sold"),
                      x="Units Sold", y="Category", orientation="h",
                      color="Units Sold",
                      color_continuous_scale="Viridis",
                      template="plotly_white")
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Regional Distribution</div>', unsafe_allow_html=True)
        reg_sales = df.groupby("Region")["Units Sold"].sum().reset_index()
        fig3 = px.pie(reg_sales, names="Region", values="Units Sold",
                      color_discrete_sequence=px.colors.qualitative.Pastel,
                      hole=0.45, template="plotly_white")
        fig3.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-header">Demand Matrix: Season vs Category</div>', unsafe_allow_html=True)
    heat = df.pivot_table(values="Units Sold", index="Seasonality", columns="Category", aggfunc="mean")
    fig4 = px.imshow(heat, text_auto=".0f", aspect="auto",
                     color_continuous_scale="RdYlGn",
                     template="plotly_white")
    fig4.update_layout(xaxis_title="Category", yaxis_title="Season")
    st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE 3 — CONSUMER CLUSTERING
# ══════════════════════════════════════════════
elif page == "Consumer Clustering Insights":
    st.markdown('<div class="section-header">Consumer Behaviour Clustering</div>', unsafe_allow_html=True)
    st.info("Clustering groups similar purchase patterns to reveal consumer behaviour segments using KMeans on key behavioural features.")

    cluster_features = [
        "Units Sold", "Price", "Discount", "Holiday/Promotion",
        "Competitor Pricing", "StockUtilization", "Inventory Level"
    ]
    n_clusters = st.slider("Number of Consumer Segments (K)", 2, 8, 4)

    @st.cache_data
    def run_clustering(n_k):
        sample = df_enc.sample(min(10000, len(df_enc)), random_state=42)
        X_clust = sample[cluster_features].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clust)
        km = KMeans(n_clusters=n_k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)
        result = sample.copy()
        result["Cluster"] = labels
        result["PCA1"] = coords[:, 0]
        result["PCA2"] = coords[:, 1]
        return result, km.inertia_

    result_df, inertia = run_clustering(n_clusters)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.scatter(result_df, x="PCA1", y="PCA2",
                         color=result_df["Cluster"].astype(str),
                         hover_data=["Category", "Region", "Units Sold", "Price"],
                         template="plotly_white",
                         title="Consumer Segments (PCA 2D Projection)",
                         color_discrete_sequence=px.colors.qualitative.Bold,
                         opacity=0.7)
        fig.update_traces(marker_size=5)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Cluster Summary**")
        summary = result_df.groupby("Cluster")[["Units Sold", "Price", "Discount", "StockUtilization"]].mean().round(2)
        for i, row in summary.iterrows():
            st.markdown(f"""<div class="algo-card">
            <b>Segment {i}</b><br>
            Units Sold: <b>{row['Units Sold']:.1f}</b><br>
            Avg Price: <b>${row['Price']:.2f}</b><br>
            Discount: <b>{row['Discount']:.1f}%</b><br>
            Stock Util: <b>{row['StockUtilization']:.2f}</b>
            </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        cluster_cat = result_df.groupby(["Cluster", "Category"]).size().reset_index(name="Count")
        cluster_cat["Cluster"] = cluster_cat["Cluster"].astype(str)
        fig = px.bar(cluster_cat, x="Cluster", y="Count", color="Category",
                     barmode="stack", template="plotly_white",
                     title="Category Distribution by Segment",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        cluster_region = result_df.groupby(["Cluster", "Region"]).size().reset_index(name="Count")
        cluster_region["Cluster"] = cluster_region["Cluster"].astype(str)
        fig = px.bar(cluster_region, x="Cluster", y="Count", color="Region",
                     barmode="group", template="plotly_white",
                     title="Region Distribution by Segment",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Cluster Profile Analysis</div>', unsafe_allow_html=True)
    radar_features = ["Units Sold", "Price", "Discount", "Competitor Pricing", "StockUtilization"]
    cluster_profile = result_df.groupby("Cluster")[radar_features].mean()
    cluster_norm = (cluster_profile - cluster_profile.min()) / (cluster_profile.max() - cluster_profile.min() + 1e-9)

    fig = go.Figure()
    colors = px.colors.qualitative.Bold
    for i, row in cluster_norm.iterrows():
        vals = row.tolist() + [row.tolist()[0]]
        cats = radar_features + [radar_features[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats, fill='toself',
            name=f"Segment {i}", line_color=colors[i % len(colors)], opacity=0.7
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                      template="plotly_white", title="Normalized Cluster Profiles")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Optimal K Evaluation (Elbow Method)</div>', unsafe_allow_html=True)
    @st.cache_data
    def elbow_data():
        sample = df_enc.sample(5000, random_state=42)
        X_s = StandardScaler().fit_transform(sample[cluster_features].fillna(0))
        inertias = []
        for k in range(2, 11):
            km = KMeans(n_clusters=k, random_state=42, n_init=5)
            km.fit(X_s)
            inertias.append({"K": k, "Inertia": km.inertia_})
        return pd.DataFrame(inertias)

    elbow_df = elbow_data()
    fig = px.line(elbow_df, x="K", y="Inertia", markers=True,
                  template="plotly_white", title="Elbow Curve — KMeans",
                  color_discrete_sequence=["#3498db"])
    fig.update_traces(marker_size=8, line_width=3)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════
# PAGE 4 — DEMAND FORECASTING
# ══════════════════════════════════════════════
elif page == "Demand Forecasting Models":
    st.markdown('<div class="section-header">Demand Forecasting — Model Training & Evaluation</div>', unsafe_allow_html=True)

    sample_size = st.slider("Training Sample Size", 5000, min(50000, len(df)), 20000, step=5000)
    use_ensemble = st.checkbox("Enable Ensemble Models", value=True)

    if st.button("Execute Training Pipeline"):
        with st.spinner("Training models..."):
            idx = np.random.choice(len(X), sample_size, replace=False)
            X_s, y_s = X.iloc[idx], y.iloc[idx]
            X_train, X_test, y_train, y_test = train_test_split(X_s, y_s, test_size=0.2, random_state=42)

            models = {
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
                "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
                "LightGBM": lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
                "Ridge Regression": Ridge(alpha=1.0),
                "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
            }

            if use_ensemble:
                base = [
                    ("xgb", xgb.XGBRegressor(n_estimators=80, random_state=42, verbosity=0)),
                    ("lgb", lgb.LGBMRegressor(n_estimators=80, random_state=42, verbose=-1)),
                    ("rf",  RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=-1)),
                ]
                models["Voting Ensemble"] = VotingRegressor(base)
                models["Stacking Ensemble"] = StackingRegressor(
                    estimators=base, final_estimator=Ridge(), cv=3
                )

            results = []
            preds_dict = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                preds_dict[name] = pred
                rmse  = np.sqrt(mean_squared_error(y_test, pred))
                mae   = mean_absolute_error(y_test, pred)
                r2    = r2_score(y_test, pred)
                mape  = mean_absolute_percentage_error(y_test, pred) * 100
                results.append({"Model": name, "RMSE": rmse, "MAE": mae,
                                 "R² Score": r2, "MAPE (%)": mape})

            res_df = pd.DataFrame(results).sort_values("R² Score", ascending=False)
            st.session_state["results"] = res_df
            st.session_state["preds_dict"] = preds_dict
            st.session_state["y_test"] = y_test.values
            best_name = res_df.iloc[0]["Model"]
            st.session_state["best_model_name"] = best_name

            tree_models = {k: v for k, v in models.items()
                           if hasattr(v, "feature_importances_")}
            if tree_models:
                best_tree_name = res_df[res_df["Model"].isin(tree_models.keys())].iloc[0]["Model"]
                st.session_state["feat_imp"] = pd.DataFrame({
                    "Feature": feature_cols,
                    "Importance": tree_models[best_tree_name].feature_importances_
                }).sort_values("Importance", ascending=False)

        st.success("Training complete.")

    if "results" in st.session_state:
        res_df = st.session_state["results"]
        preds_dict = st.session_state["preds_dict"]
        y_test_arr = st.session_state["y_test"]

        st.markdown('<div class="section-header">Model Accuracy Comparison</div>', unsafe_allow_html=True)
        display_df = res_df.copy()
        display_df["R² Score"] = display_df["R² Score"].map("{:.4f}".format)
        display_df["RMSE"] = display_df["RMSE"].map("{:.2f}".format)
        display_df["MAE"] = display_df["MAE"].map("{:.2f}".format)
        display_df["MAPE (%)"] = display_df["MAPE (%)"].map("{:.2f}".format)
        st.dataframe(display_df.reset_index(drop=True), use_container_width=True)

        fig = px.bar(res_df.sort_values("R² Score"),
                     x="R² Score", y="Model", orientation="h",
                     color="R² Score", color_continuous_scale="RdYlGn",
                     template="plotly_white",
                     title="Model R² Score Comparison",
                     text="R² Score")
        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig.update_layout(xaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        metrics_long = res_df.melt(id_vars="Model", value_vars=["RMSE", "MAE"],
                                    var_name="Metric", value_name="Value")
        fig2 = px.bar(metrics_long, x="Model", y="Value", color="Metric",
                      barmode="group", template="plotly_white",
                      title="RMSE vs MAE by Model",
                      color_discrete_sequence=["#3498db", "#e67e22"])
        fig2.update_layout(xaxis_tickangle=-20)
        st.plotly_chart(fig2, use_container_width=True)

        best = st.session_state["best_model_name"]
        st.markdown(f'<div class="section-header">Actual vs Predicted — {best}</div>', unsafe_allow_html=True)
        sample_idx = np.random.choice(len(y_test_arr), min(300, len(y_test_arr)), replace=False)
        actual   = y_test_arr[sample_idx]
        predicted = preds_dict[best][sample_idx]

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=actual, y=predicted, mode='markers',
                                   marker=dict(color='#3498db', opacity=0.5, size=5),
                                   name='Predictions'))
        lim = max(actual.max(), predicted.max())
        fig3.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode='lines',
                                   line=dict(color='red', dash='dash'),
                                   name='Perfect Fit'))
        fig3.update_layout(xaxis_title="Actual", yaxis_title="Predicted",
                            template="plotly_white",
                            title=f"Actual vs Predicted — {best}")
        st.plotly_chart(fig3, use_container_width=True)

        residuals = actual - predicted
        fig4 = px.histogram(pd.DataFrame({"Residual": residuals}), x="Residual",
                             nbins=50, template="plotly_white",
                             color_discrete_sequence=["#2c3e50"],
                             title="Residual Distribution")
        st.plotly_chart(fig4, use_container_width=True)

        if "feat_imp" in st.session_state:
            st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
            fi = st.session_state["feat_imp"].head(15)
            fig5 = px.bar(fi, x="Importance", y="Feature", orientation="h",
                          color="Importance", color_continuous_scale="Blues",
                          template="plotly_white", title="Top 15 Features by Importance")
            st.plotly_chart(fig5, use_container_width=True)

# ══════════════════════════════════════════════
# PAGE 5 — PREDICTION
# ══════════════════════════════════════════════
elif page == "Real-Time Prediction Engine":
    st.markdown('<div class="section-header">Predict Product Demand</div>', unsafe_allow_html=True)
    st.info("Enter product and market details below to get an instant demand forecast using a pre-trained ensemble model.")

    @st.cache_resource
    def get_prediction_model():
        sample_idx = np.random.choice(len(X), min(30000, len(X)), replace=False)
        X_s, y_s = X.iloc[sample_idx], y.iloc[sample_idx]
        base = [
            ("xgb", xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)),
            ("lgb", lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)),
            ("rf",  RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ]
        model = VotingRegressor(base)
        model.fit(X_s, y_s)
        return model

    with st.spinner("Loading prediction model..."):
        pred_model = get_prediction_model()

    st.markdown("#### Input Parameters")

    cat_map = {"Groceries": 0, "Toys": 1, "Clothing": 2, "Electronics": 3, "Furniture": 4}
    reg_map = {"North": 2, "South": 3, "East": 0, "West": 4}
    seas_map = {"Spring": 2, "Summer": 3, "Winter": 1, "Autumn": 0}
    weather_map = {"Sunny": 3, "Rainy": 1, "Snowy": 2, "Cloudy": 0}

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Product Information**")
        category = st.selectbox("Category", list(cat_map.keys()))
        price = st.number_input("Price ($)", min_value=1.0, max_value=200.0, value=45.0, step=0.5)
        discount = st.slider("Discount (%)", 0, 50, 10)
        competitor_price = st.number_input("Competitor Price ($)", min_value=1.0, max_value=200.0, value=48.0, step=0.5)

    with col2:
        st.markdown("**Store & Market Context**")
        region = st.selectbox("Region", list(reg_map.keys()))
        inventory_level = st.number_input("Current Inventory Level", min_value=50, max_value=500, value=200)
        units_ordered = st.number_input("Units Ordered", min_value=0, max_value=500, value=80)
        holiday = st.selectbox("Holiday/Promotion Active?", ["No", "Yes"])

    with col3:
        st.markdown("**Environmental Factors**")
        seasonality = st.selectbox("Season", list(seas_map.keys()))
        weather = st.selectbox("Weather Condition", list(weather_map.keys()))
        month = st.slider("Month", 1, 12, 6)
        day_of_week = st.slider("Day of Week (0=Mon)", 0, 6, 2)

    quarter = (month - 1) // 3 + 1
    week_of_year = min(month * 4, 52)
    price_discounted = price * (1 - discount / 100)
    price_diff = price - competitor_price
    stock_util = 0.5
    holiday_val = 1 if holiday == "Yes" else 0

    input_dict = {
        "Inventory Level": inventory_level,
        "Units Ordered": units_ordered,
        "Price": price,
        "Discount": discount,
        "Holiday/Promotion": holiday_val,
        "Competitor Pricing": competitor_price,
        "Month": month,
        "DayOfWeek": day_of_week,
        "Quarter": quarter,
        "WeekOfYear": week_of_year,
        "PriceDiscounted": price_discounted,
        "PriceDiff": price_diff,
        "StockUtilization": stock_util,
        "Category_enc": cat_map[category],
        "Region_enc": reg_map[region],
        "Seasonality_enc": seas_map[seasonality],
        "Weather Condition_enc": weather_map[weather],
    }

    input_df = pd.DataFrame([input_dict])[feature_cols]

    if st.button("Generate Forecast"):
        prediction = pred_model.predict(input_df)[0]
        prediction = max(0, round(prediction, 1))

        st.markdown(f"""
        <div class="pred-result">
            <p>Predicted Units Sold</p>
            <h2>{prediction:.0f} units</h2>
            <p>{category} | {region} | {seasonality} | {weather}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Price Sensitivity Analysis</div>', unsafe_allow_html=True)
        prices = np.linspace(max(1, price * 0.5), price * 1.8, 30)
        sens_preds = []
        for p in prices:
            row = input_dict.copy()
            row["Price"] = p
            row["PriceDiscounted"] = p * (1 - discount / 100)
            row["PriceDiff"] = p - competitor_price
            sens_preds.append(pred_model.predict(pd.DataFrame([row])[feature_cols])[0])

        fig = px.line(x=prices, y=sens_preds,
                      labels={"x": "Price ($)", "y": "Predicted Units Sold"},
                      template="plotly_white",
                      title="Demand Sensitivity to Price",
                      color_discrete_sequence=["#3498db"])
        fig.add_vline(x=price, line_dash="dash", line_color="red",
                      annotation_text=f"Current: ${price:.2f}")
        fig.update_traces(line_width=3)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Discount Sensitivity Analysis</div>', unsafe_allow_html=True)
        discounts = np.arange(0, 51, 5)
        disc_preds = []
        for d in discounts:
            row = input_dict.copy()
            row["Discount"] = d
            row["PriceDiscounted"] = price * (1 - d / 100)
            disc_preds.append(pred_model.predict(pd.DataFrame([row])[feature_cols])[0])

        fig2 = px.bar(x=discounts, y=disc_preds,
                      labels={"x": "Discount (%)", "y": "Predicted Units Sold"},
                      template="plotly_white",
                      title="Demand Sensitivity to Discount",
                      color=disc_preds, color_continuous_scale="Greens")
        fig2.add_vline(x=discount, line_dash="dash", line_color="red",
                       annotation_text=f"Current: {discount}%")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-header">Input Summary</div>', unsafe_allow_html=True)
        labels = ["Price", "Discount", "Inventory", "Competitor Price", "Units Ordered"]
        vals   = [price, discount, inventory_level, competitor_price, units_ordered]
        maxs   = [200, 50, 500, 200, 500]
        norm   = [v / m for v, m in zip(vals, maxs)]

        fig3 = go.Figure()
        for i, (lbl, nv, rv) in enumerate(zip(labels, norm, vals)):
            fig3.add_trace(go.Bar(
                x=[lbl], y=[nv],
                name=lbl,
                text=[f"{rv:.1f}"],
                textposition="outside",
                marker_color=px.colors.qualitative.Bold[i],
            ))
        fig3.update_layout(showlegend=False, template="plotly_white",
                           title="Normalized Input Values",
                           yaxis=dict(range=[0, 1.3]))
        st.plotly_chart(fig3, use_container_width=True)