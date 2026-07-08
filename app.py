"""
Gold Price Prediction Dashboard
--------------------------------
A professional, interactive Streamlit application that predicts gold prices
using a machine learning model trained with cross-validation on macroeconomic
indicators. Includes live prediction, historical data explorer, correlation
analysis, and model performance / cross-validation reporting.

Run with:  streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Gold Price Predictor | ML Dashboard",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Custom CSS — professional "gold" themed UI
# --------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&family=Playfair+Display:wght@600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    .main {
        background: linear-gradient(180deg, #0f0f0f 0%, #1a1a1a 100%);
    }

    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #D4AF37, #F4E5B2, #D4AF37);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
        padding-top: 0.5rem;
    }

    .hero-subtitle {
        text-align: center;
        color: #B0B0B0;
        font-size: 1.05rem;
        margin-top: 0.2rem;
        margin-bottom: 1.5rem;
        letter-spacing: 0.5px;
    }

    .metric-card {
        background: linear-gradient(135deg, #1f1f1f, #2a2a2a);
        border: 1px solid #D4AF37;
        border-radius: 14px;
        padding: 1.1rem 1rem;
        text-align: center;
        box-shadow: 0 4px 14px rgba(212, 175, 55, 0.15);
    }

    .metric-label {
        color: #B0B0B0;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-value {
        color: #D4AF37;
        font-size: 1.7rem;
        font-weight: 700;
        font-family: 'Playfair Display', serif;
    }

    .prediction-box {
        background: linear-gradient(135deg, #2b2412, #1a1a1a 60%);
        border: 2px solid #D4AF37;
        border-radius: 18px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(212, 175, 55, 0.25);
    }

    .prediction-value {
        font-family: 'Playfair Display', serif;
        font-size: 3.2rem;
        font-weight: 700;
        color: #F4E5B2;
    }

    .section-header {
        font-family: 'Playfair Display', serif;
        color: #D4AF37;
        border-bottom: 2px solid #D4AF37;
        padding-bottom: 0.4rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #141414, #1c1c1c);
        border-right: 1px solid #D4AF37;
    }

    .stButton>button {
        background: linear-gradient(90deg, #D4AF37, #B8860B);
        color: #111;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        transition: 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 14px rgba(212, 175, 55, 0.4);
    }

    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Cached data / model loading
# --------------------------------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    model = joblib.load(BASE_DIR / "gold_price_model.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    with open(BASE_DIR / "model_metadata.json") as f:
        metadata = json.load(f)
    return model, scaler, metadata


@st.cache_data
def load_data():
    df = pd.read_csv(BASE_DIR / "gold_price_data.csv", parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


@st.cache_data
def load_comparison_results():
    return pd.read_csv(BASE_DIR / "model_comparison_results.csv")


model, scaler, metadata = load_model_artifacts()
df = load_data()
comparison_df = load_comparison_results()
FEATURES = metadata["features"]

# --------------------------------------------------------------------------
# Hero header
# --------------------------------------------------------------------------
st.markdown('<div class="hero-title">🏆 Gold Price Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Machine Learning powered forecasting using macroeconomic '
    'indicators &nbsp;•&nbsp; Trained with K-Fold & Time-Series Cross-Validation</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Top KPI strip
# --------------------------------------------------------------------------
latest = df.iloc[-1]
prev = df.iloc[-2]
delta = latest["Gold_Price_USD"] - prev["Gold_Price_USD"]
delta_pct = (delta / prev["Gold_Price_USD"]) * 100

k1, k2, k3, k4, k5 = st.columns(5)
kpi_data = [
    (k1, "Latest Gold Price", f"${latest['Gold_Price_USD']:,.2f}"),
    (k2, "Daily Change", f"{delta:+.2f} ({delta_pct:+.2f}%)"),
    (k3, "Model Used", metadata["best_model_name"]),
    (k4, "Test R² Score", f"{metadata['test_r2']:.4f}"),
    (k5, "Test MAE", f"${metadata['test_mae']:.2f}"),
]
for col, label, value in kpi_data:
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# --------------------------------------------------------------------------
# Sidebar — prediction inputs
# --------------------------------------------------------------------------
st.sidebar.markdown("## 🔮 Predict Gold Price")
st.sidebar.markdown("Adjust the market indicators below to generate a live prediction.")

fmin = metadata["feature_min"]
fmax = metadata["feature_max"]
fmed = metadata["feature_medians"]

friendly_labels = {
    "Silver_Price_USD": "Silver Price (USD/oz)",
    "Crude_Oil_USD": "Crude Oil Price (USD/bbl)",
    "USD_Index": "US Dollar Index (DXY)",
    "SP500_Index": "S&P 500 Index",
    "Inflation_Rate_%": "Inflation Rate (% YoY)",
    "Interest_Rate_%": "Fed Interest Rate (%)",
    "VIX_Volatility_Index": "VIX Volatility Index",
    "USD_INR_Rate": "USD/INR Exchange Rate",
    "Gold_ETF_Holdings_Tonnes": "Gold ETF Holdings (Tonnes)",
    "Gold_Trading_Volume": "Gold Trading Volume",
    "Gold_Price_Lag1": "Gold Price — 1 Day Ago (USD)",
    "Gold_Price_Lag7": "Gold Price — 7 Days Ago (USD)",
    "Gold_MA_7": "Gold 7-Day Moving Avg (USD)",
    "Gold_MA_30": "Gold 30-Day Moving Avg (USD)",
    "Gold_Volatility_7": "Gold 7-Day Volatility (Std Dev)",
}

with st.sidebar.expander("📌 Quick Presets", expanded=False):
    preset = st.radio(
        "Load a market scenario:",
        ["Current Market (latest data)", "High Inflation Scenario", "Market Crash / Risk-Off", "Custom"],
        index=0,
    )

user_inputs = {}
preset_overrides = {}
if preset == "High Inflation Scenario":
    preset_overrides = {"Inflation_Rate_%": fmax["Inflation_Rate_%"] * 0.85, "VIX_Volatility_Index": fmed["VIX_Volatility_Index"] * 1.3}
elif preset == "Market Crash / Risk-Off":
    preset_overrides = {"VIX_Volatility_Index": fmax["VIX_Volatility_Index"] * 0.8, "SP500_Index": fmin["SP500_Index"] * 1.15}

st.sidebar.markdown("---")
for feat in FEATURES:
    default_val = latest[feat] if preset == "Current Market (latest data)" else preset_overrides.get(feat, fmed[feat])
    step = round((fmax[feat] - fmin[feat]) / 200, 3) or 0.01
    user_inputs[feat] = st.sidebar.slider(
        friendly_labels.get(feat, feat),
        min_value=float(fmin[feat]),
        max_value=float(fmax[feat]),
        value=float(np.clip(default_val, fmin[feat], fmax[feat])),
        step=float(step),
    )

predict_clicked = st.sidebar.button("🔍 Predict Gold Price", use_container_width=True)

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Live Prediction", "📈 Historical Data", "🔗 Correlation Analysis",
    "🧪 Model & Cross-Validation", "ℹ️ About Project"
])

# ---------------------------- TAB 1: Prediction ----------------------------
with tab1:
    left, right = st.columns([1.1, 1.4])

    with left:
        st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)
        input_df = pd.DataFrame([user_inputs])[FEATURES]
        input_scaled = scaler.transform(input_df.values)
        prediction = model.predict(input_scaled)[0]

        st.markdown(f"""
        <div class="prediction-box">
            <div style="color:#B0B0B0; letter-spacing:1px; text-transform:uppercase; font-size:0.85rem;">
                Predicted Gold Price
            </div>
            <div class="prediction-value">${prediction:,.2f}</div>
            <div style="color:#888; margin-top:0.5rem;">per troy ounce (USD)</div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        margin = metadata["test_mae"]
        st.info(
            f"Estimated confidence range: **${prediction - margin:,.2f} — ${prediction + margin:,.2f}** "
            f"(± test MAE of the {metadata['best_model_name']} model)"
        )

        diff_vs_latest = prediction - latest["Gold_Price_USD"]
        direction = "📈 above" if diff_vs_latest > 0 else "📉 below"
        st.markdown(f"This prediction is **{abs(diff_vs_latest):.2f} USD {direction}** the latest recorded price of **${latest['Gold_Price_USD']:,.2f}**.")

    with right:
        st.markdown('<div class="section-header">Your Input vs Historical Range</div>', unsafe_allow_html=True)
        gauge_features = ["Silver_Price_USD", "USD_Index", "Interest_Rate_%", "Inflation_Rate_%", "VIX_Volatility_Index"]
        fig = go.Figure()
        for i, feat in enumerate(gauge_features):
            norm_val = (user_inputs[feat] - fmin[feat]) / (fmax[feat] - fmin[feat] + 1e-9)
            fig.add_trace(go.Bar(
                y=[friendly_labels.get(feat, feat)],
                x=[norm_val],
                orientation="h",
                marker=dict(color="#D4AF37"),
                showlegend=False,
                text=[f"{user_inputs[feat]:.2f}"],
                textposition="outside",
            ))
        fig.update_layout(
            xaxis=dict(range=[0, 1.15], title="Position within historical min–max range", showgrid=False),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Feature Importance Driving This Model</div>', unsafe_allow_html=True)
        fi_df = pd.DataFrame(metadata["feature_importance"]).head(8)
        fig2 = px.bar(
            fi_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale=["#5a4a1a", "#D4AF37", "#F4E5B2"],
        )
        fig2.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(autorange="reversed"), height=320, showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

# ---------------------------- TAB 2: Historical Data ----------------------------
with tab2:
    st.markdown('<div class="section-header">Historical Gold Price Trend</div>', unsafe_allow_html=True)

    date_range = st.slider(
        "Select date range",
        min_value=df["Date"].min().to_pydatetime(),
        max_value=df["Date"].max().to_pydatetime(),
        value=(df["Date"].min().to_pydatetime(), df["Date"].max().to_pydatetime()),
        format="YYYY-MM-DD",
    )
    filtered = df[(df["Date"] >= date_range[0]) & (df["Date"] <= date_range[1])]

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=filtered["Date"], y=filtered["Gold_Price_USD"],
        mode="lines", name="Gold Price", line=dict(color="#D4AF37", width=1.6),
        fill="tozeroy", fillcolor="rgba(212,175,55,0.08)",
    ))
    fig3.add_trace(go.Scatter(
        x=filtered["Date"], y=filtered["Gold_MA_30"],
        mode="lines", name="30-Day Moving Avg", line=dict(color="#8C8C8C", width=1.2, dash="dot"),
    ))
    fig3.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=420, xaxis_title="Date", yaxis_title="Gold Price (USD)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig3, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Gold vs Silver</div>', unsafe_allow_html=True)
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=filtered["Date"], y=filtered["Gold_Price_USD"], name="Gold", line=dict(color="#D4AF37")))
        fig4.add_trace(go.Scatter(x=filtered["Date"], y=filtered["Silver_Price_USD"] * 40, name="Silver (x40 scaled)", line=dict(color="#C0C0C0")))
        fig4.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320)
        st.plotly_chart(fig4, use_container_width=True)
    with c2:
        st.markdown('<div class="section-header">Gold Trading Volume</div>', unsafe_allow_html=True)
        fig5 = px.bar(filtered.iloc[::15], x="Date", y="Gold_Trading_Volume", color_discrete_sequence=["#D4AF37"])
        fig5.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320)
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown('<div class="section-header">Raw Data Explorer</div>', unsafe_allow_html=True)
    st.dataframe(filtered.tail(200).sort_values("Date", ascending=False), use_container_width=True, height=320)

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered data as CSV", csv_bytes, "gold_price_filtered.csv", "text/csv")

# ---------------------------- TAB 3: Correlation ----------------------------
with tab3:
    st.markdown('<div class="section-header">Correlation Between Gold Price & Market Indicators</div>', unsafe_allow_html=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric_cols].corr()

    fig6 = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdYlGn", aspect="auto",
        zmin=-1, zmax=1,
    )
    fig6.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=600)
    st.plotly_chart(fig6, use_container_width=True)

    st.markdown('<div class="section-header">Explore Any Two Variables</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    x_var = c1.selectbox("X-axis variable", numeric_cols, index=numeric_cols.index("USD_Index"))
    y_var = c2.selectbox("Y-axis variable", numeric_cols, index=numeric_cols.index("Gold_Price_USD"))
    fig7 = px.scatter(df, x=x_var, y=y_var, color="Gold_Price_USD", color_continuous_scale="YlOrBr", opacity=0.55)
    fig7.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=450)
    st.plotly_chart(fig7, use_container_width=True)

# ---------------------------- TAB 4: Model & CV ----------------------------
with tab4:
    st.markdown('<div class="section-header">Model Comparison (7 Algorithms Trained)</div>', unsafe_allow_html=True)
    st.dataframe(comparison_df.style.background_gradient(cmap="YlOrBr", subset=["Test_R2"]), use_container_width=True)

    st.markdown('<div class="section-header">K-Fold vs Time-Series Cross-Validation</div>', unsafe_allow_html=True)
    st.markdown("""
    Two cross-validation strategies were used during training:
    - **K-Fold CV (shuffled, 5 splits):** randomly shuffles data — can leak future
      information into training folds for time-series data, often *inflating* scores.
    - **TimeSeriesSplit CV (5 splits):** always trains on the past and validates on a
      later block, respecting chronological order — the statistically correct
      approach for forecasting problems.
    """)

    fig8 = go.Figure()
    fig8.add_trace(go.Bar(x=comparison_df["Model"], y=comparison_df["KFold_CV_R2_Mean"], name="KFold CV R²", marker_color="#4C72B0"))
    fig8.add_trace(go.Bar(x=comparison_df["Model"], y=comparison_df["TimeSeriesCV_R2_Mean"], name="TimeSeriesSplit CV R²", marker_color="#D4AF37"))
    fig8.update_layout(
        barmode="group", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=420, yaxis_title="R² Score",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig8, use_container_width=True)

    st.warning(
        "⚠️ Notice how tree-based ensembles (Random Forest, Gradient Boosting, XGBoost) score "
        "near-perfectly under shuffled K-Fold CV, yet perform poorly on genuinely unseen future "
        "prices (TimeSeriesSplit / chronological test set). This is because gold price has a strong "
        "long-term trend, and tree models cannot extrapolate beyond the price range seen in training. "
        f"The **{metadata['best_model_name']}** model was selected because it generalizes best to real "
        "future forecasting, not just because of a high average CV score."
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Selected Model", metadata["best_model_name"])
    m2.metric("Test R² (chronological)", f"{metadata['test_r2']:.4f}")
    m3.metric("TimeSeriesSplit CV R² (mean ± std)", f"{metadata['tscv_cv_r2_mean']:.3f} ± {metadata['tscv_cv_r2_std']:.3f}")

# ---------------------------- TAB 5: About ----------------------------
with tab5:
    st.markdown('<div class="section-header">About This Project</div>', unsafe_allow_html=True)
    st.markdown(f"""
This dashboard predicts **daily gold prices (USD/oz)** using a machine learning
regression model trained on **{len(df):,} rows** of daily data spanning
**{metadata['data_min_date']} to {metadata['data_max_date']}**.

**Model:** {metadata['best_model_name']}
**Features used:** {len(FEATURES)} engineered market indicators (macro drivers + lag/rolling price features)
**Validation:** 5-fold K-Fold CV **and** 5-split TimeSeriesSplit CV, plus a held-out chronological test set

**Why this is useful:**
- Investors and analysts can explore how macro shifts (inflation, interest rates, USD strength) affect gold prices
- Demonstrates a real pitfall of naive cross-validation on time-series/financial data
- Fully interactive: adjust any indicator and get an instant price prediction with a confidence range

**Tech stack:** Python, scikit-learn, XGBoost, pandas, Streamlit, Plotly
""")
    st.markdown('<div class="section-header">Project Files</div>', unsafe_allow_html=True)
    st.code("""
gold_price_project/
├── app.py                          # This Streamlit application
├── gold_price_data.csv             # 10-year daily dataset (3,600+ rows)
├── Gold_Price_Prediction.ipynb     # Full training notebook (EDA + CV + model selection)
├── gold_price_model.pkl            # Trained best model (serialized)
├── scaler.pkl                      # Fitted StandardScaler for input features
├── model_metadata.json             # Feature list, metrics, min/max/median values
├── model_comparison_results.csv    # Performance table of all 7 models
└── requirements.txt                # Python dependencies
    """, language="text")

st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#666; font-size:0.85rem;">'
    'Gold Price Prediction Dashboard • Built with Streamlit & Scikit-learn • For educational purposes only, not financial advice.'
    '</div>', unsafe_allow_html=True
)
