# 🏆 Gold Price Prediction Dashboard

An end-to-end machine learning project that predicts daily gold prices (USD/oz)
using macroeconomic indicators, trained with **K-Fold** and **TimeSeriesSplit**
cross-validation, deployed as an interactive **Streamlit** dashboard.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit dashboard (run this) |
| `gold_price_data.csv` | 10-year daily dataset (3,621 rows, 17 columns) |
| `Gold_Price_Prediction.ipynb` | Full training notebook: EDA, feature engineering, cross-validation, model selection |
| `gold_price_model.pkl` | Trained model, ready to load with `joblib` |
| `scaler.pkl` | Fitted `StandardScaler` — must be applied to new inputs before prediction |
| `model_metadata.json` | Feature list, performance metrics, min/max/median values |
| `model_comparison_results.csv` | Performance table comparing all 7 trained algorithms |
| `requirements.txt` | Python dependencies |
| `generate_data.py` | Script used to generate the dataset |
| `train_model.py` | Script used to train and save the model (mirrors the notebook) |
| `build_notebook.py` | Script that programmatically generated the `.ipynb` |

See the full chat response for a detailed explanation of every file.
