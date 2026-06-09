# Changelog

All notable changes to this project are documented here.

## [1.0.0] — 2024

### Added
- **Dashboard Overview** — KPI metric cards, monthly sales area chart, category bar chart, regional pie chart, season × category demand heatmap
- **Consumer Clustering Insights** — KMeans segmentation with adjustable K (2–8), PCA 2D scatter projection, per-segment summary cards, category/region stacked bars, normalized radar chart, elbow curve
- **Demand Forecasting Models** — training pipeline for 8 models (Random Forest, Gradient Boosting, XGBoost, LightGBM, Ridge, Decision Tree, Voting Ensemble, Stacking Ensemble); evaluation table with RMSE, MAE, R², MAPE; actual vs predicted scatter; residual histogram; feature importance chart
- **Real-Time Prediction Engine** — Voting Ensemble (XGBoost + LightGBM + RF) pre-trained for live inference; price sensitivity line chart; discount sensitivity bar chart; normalized input summary
- Feature engineering pipeline: `PriceDiscounted`, `PriceDiff`, `StockUtilization`, temporal features
- Dark-themed sidebar with dataset metadata
- Streamlit `@st.cache_data` and `@st.cache_resource` for performance
