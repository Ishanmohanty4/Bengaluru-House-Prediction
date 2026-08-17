# 🏡 Bengaluru House Price Prediction & Web App

An end-to-end Machine Learning web application to predict residential real estate prices across Bengaluru, India based on location, square footage, bedroom count (BHK), and bathroom count.

🔗 **Live Streamlit App:** [Bengaluru House Price Estimator](https://bengaluru-house-predictior.streamlit.app/)

---

## 📌 Project Overview
Estimating property prices in Bengaluru is challenging due to high variance across tech corridors, micro-markets, and property sizes. This project builds a machine learning pipeline that handles heavy target skewness, cleans unstandardized textual unit data, removes domain-specific outliers, and serves predictions via an interactive Streamlit UI.

---

## ⚙️ Key Workflow & Engineering

1. **Data Cleaning & Standardization:**
   - Handled missing values across critical features.
   - Parsed non-standard `total_sqft` entries (e.g., averaged range values like `'2100 - 2850'` and converted non-sqft units).
   - Extracted numerical `size` counts from categorical size strings.

2. **Outlier Filtering & Feature Engineering:**
   - Enforced standard architectural threshold: removed entries with `< 300 sqft / Size`.
   - Dimension reduction: grouped rare localities ($\le 10$ records) into an `'other'` bucket.
   - Removed location-wise price per sqft outliers ($\pm 1\sigma$).
   - Filtered out Size pricing anomalies and excessive bathroom counts (`bath > size + 2`).

3. **Target Transformation:**
   - Log-transformed (`np.log1p`) the right-skewed `price` column to normalize residuals and stabilize regression variance.

---

## 📊 Model Evaluation Benchmark

Evaluated on test data (metrics mapped back to original scale in ₹ Lakhs):

| Model | $R^2$ Score | MAE (₹ Lakhs) | RMSE (₹ Lakhs) |
| :--- | :--- | :--- | :--- |
| **Random Forest Regressor** | **0.8571** | **₹16.07** | **₹32.63** |
| **Gradient Boosting Regressor** | 0.8473 | ₹17.79 | ₹33.74 |
| **Linear Regression** | 0.7666 | ₹19.28 | ₹41.71 |
| **Ridge Regression** | 0.7543 | ₹19.49 | ₹42.80 |
| **Lasso Regression** | 0.5633 | ₹24.30 | ₹57.06 |

> **Final Selected Model:** **Random Forest Regressor** achieved the best predictive performance with an $R^2$ of ~0.8571 and lowest average error.

---

## 📁 Repository Structure

```text
├── app.py                             # Streamlit user interface & inference logic
├── bangalore_house_price_model.pickle # Trained Random Forest model artifact
├── columns.json                       # One-hot encoded feature schema
├── Bengaluru_House_Data.csv           # Raw dataset
├── requirements.txt                   # Environment dependencies
└── README.md                          # Project documentation
