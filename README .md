# Online News Popularity Prediction Model

---

**Name:** **Jain Prasannakumar**

**Organization:** **Entri Elevate**

**Date:** **06-10-2026**

---

## 1. Overview of Problem Statement

The number of times a news article gets shared on social media is highly uncertain and depends on many interacting factors — topic, writing style, publish timing, and keyword performance, among others. Predicting share count accurately would help publishers understand what drives engagement and make more informed editorial decisions. This project builds a regression model to predict article shares from measurable content and metadata attributes.

## 2. Objective

To develop a machine learning model that predicts the number of shares a news article will receive, based on its structural, content, and metadata features, and to deploy it through a Flask web interface for interactive predictions.

## 3. Data Description

- **Source:** UCI Machine Learning Repository — Online News Popularity Dataset (articles originally published by Mashable, 2013–2015)
- **Rows:** 39,644 articles
- **Features:** 58 predictive features (59 columns including the identifier/target), grouped as:
  - **Article structure** — `n_tokens_title`, `n_tokens_content`, `num_hrefs`, `num_imgs`, `num_videos`, `average_token_length`
  - **Vocabulary/token stats** — `n_unique_tokens`, `n_non_stop_words`, `n_non_stop_unique_tokens`
  - **Topic channel (one-hot)** — `data_channel_is_tech`, `_world`, `_entertainment`, `_bus`, `_lifestyle`, `_socmed`
  - **Publish timing (one-hot)** — `weekday_is_monday` ... `weekday_is_sunday`, `is_weekend`
  - **Keyword performance** — `kw_min_min`, `kw_max_max`, `kw_avg_avg`, etc.
  - **Sentiment/tone** — `global_subjectivity`, `global_sentiment_polarity`, `title_sentiment_polarity`, `rate_positive_words`
  - **Topic modeling** — `LDA_00` through `LDA_04`
  - **Self-reference** — `self_reference_min_shares`, `_max_shares`, `_avg_sharess`
- **Target:** `shares` (log-transformed to `log_shares` for modeling)

## 4. Data Collection

Dataset imported directly from the provided CSV (`OnlineNewsPopularity.csv`). No scraping or API collection was required — the dataset was already assembled and labeled.

## 5. Data Preprocessing - Data Cleaning

- **Missing values:** none found in the raw dataset — no imputation was necessary.
- **Outliers:**
  - Removed 1 row with an invalid `n_unique_tokens` ratio (> 1, a known data artifact).
  - Checked outliers in `shares` using the IQR method; capped extreme values at the 99th percentile for the final training pipeline to limit the leverage of viral outliers without discarding them.
- **Skewed data:** `shares` is heavily right-skewed (mean 3,395 vs. median 1,400); addressed with a `log1p` transformation, which was the single biggest factor in making the target usable for regression.
- **Column name cleanup:** renamed the raw `' shares'` column (leading whitespace in source CSV) to `'shares'`.

## 6. Exploratory Data Analysis (EDA)

- Visualized the `shares` distribution (boxplot) to confirm the right-skew and outlier pattern described above.
- Examined article counts across topic channels (tech, world, entertainment, business, lifestyle, social media).
- Checked for multicollinearity among numeric features via a correlation matrix — three feature pairs exceeded |r| > 0.9: `n_non_stop_unique_tokens`/`n_unique_tokens`, `average_token_length`/`n_non_stop_words`, and `kw_avg_min`/`kw_max_min`.

## 7. Feature Engineering

- No label/one-hot encoding was required — categorical attributes (topic channel, weekday) arrive pre-encoded as binary dummy columns in the source dataset.
- Target engineering: `log_shares = log1p(shares)`.

## 8. Feature Selection

Used two complementary algorithms to identify relevant features:
- **Random Forest feature importance** — top features included `kw_avg_avg`, `kw_max_avg`, `self_reference_avg_sharess`, `kw_avg_max`, and the `LDA_*` topic scores.
- **SelectKBest (f_regression)** — top features included `kw_avg_avg`, `data_channel_is_world`, `is_weekend`, `num_hrefs`, `num_imgs`.
- Features both methods agreed were top-ranked: `kw_avg_avg`, `kw_max_avg`, `self_reference_avg_sharess`, `self_reference_min_shares`, `LDA_02`, `LDA_03`.
- The final deployed model uses the full feature set rather than a reduced one; the reduced set is available for a future lighter-weight model comparison.

## 9. Split Data into Training and Testing Sets

80/20 train-test split (`train_test_split`, `random_state=42`) — 31,714 training rows / 7,929 test rows.

## 10. Feature Scaling

Standardization (`StandardScaler`) applied to numeric features, bundled inside each model's `Pipeline` so raw input can be passed directly at train and inference time.

## 11. Build the ML Model

Five regression algorithms were implemented and compared:

| Model | R² | RMSE | MAE |
|---|---|---|---|
| Linear Regression | 0.127 | 0.865 | 0.644 |
| Random Forest Regressor | 0.167 | 0.845 | 0.634 |
| Gradient Boosting Regressor | 0.176 | 0.840 | 0.625 |
| AdaBoost Regressor | 0.1124 | 0.8504| 0.6583 |
| SVR | 0.1294 | 0.8423 | 0.6080 |


## 12. Model Evaluation

Regression metrics used: **MAE, MSE, RMSE, R² Score**, computed on the held-out test set against `log_shares` for every model, both before and after tuning (see Section 13).

## 13. Hyperparameter Tuning

All models with tunable hyperparameters were tuned via `GridSearchCV` (cv=5, scoring='r2'). Linear Regression was intentionally excluded — its coefficients are solved analytically and it has no hyperparameters to search over. SVR was tuned on a 5,000-row subsample (full-dataset grid search on SVR is computationally impractical) and its best parameters refit on the full training set.

**Tuned results:**

| Model | R² | RMSE | MAE |
|---|---|---|---|
| Random Forest (tuned) | 0.1766 | 0.8192 | 0.6186 |
| Gradient Boosting (tuned) | 0.1764 | 0.8192 | 0.6169 |
| SVR  | 0.1294 | 0.8423 | 0.6080 |
| AdaBoost (tuned) | 0.1124 | 0.8504 | 0.6583 |

Random Forest and Gradient Boosting are effectively tied after tuning (R² difference of 0.0002, within noise), both improving over their untuned baselines. AdaBoost remained the weakest performer even after tuning. SVR showed the largest relative gain from tuning but still trails the top two.

## 14. Save the Model

Final model saved as `model.pkl` (a scikit-learn `Pipeline` containing the fitted `StandardScaler` and `GradientBoostingRegressor` together). Supporting files: `feature_columns.pkl` (exact input column order) and `feature_defaults.pkl` (median values used to auto-fill features not collected from the user-facing form).


## 15. Test with Unseen Data

Evaluated on the 20% held-out test split (see Sections 11 and 13 results). No separate external/unseen dataset has been tested yet.

## 16. Interpretation of Results (Conclusion)

The model explains a modest portion of the variance in article shares (R² ≈ 0.18 after tuning), consistent with known findings on this dataset. Key limitation: the available features describe article structure and metadata only — they do not capture external drivers of virality such as promotion, social placement, or who shares an article first. A modest R² here reflects this inherent data limitation rather than a modeling failure. Tuning produced measurable but incremental gains across all models, reinforcing that the ceiling on predictive power is set more by the available features than by model choice or configuration.

## 17. Future Work

- Retrain on the reduced, feature-selected set (from Section 8) and compare performance/interpretability against the full feature set.
- Periodically update the model as new article data becomes available.
- Deploy the Flask application to a public host (e.g. Render, Railway, PythonAnywhere) for shareable access.
- Explore MLP Regressor as an additional model for comparison.

---

## Project Structure

```
├── model.py                  # Trains and saves the final model
├── app.py                    # Flask application serving predictions
├── tuning.py                 # Hyperparameter tuning for RF, GBR, AdaBoost, SVR
├── model.pkl                 # Trained Pipeline (scaler + GradientBoostingRegressor)
├── feature_columns.pkl       # Exact column order expected by the model
├── feature_defaults.pkl      # Median values for auto-filled features
├── scaler.joblib              # Fitted StandardScaler 
├── tuned_model_comparison.csv # Tuned model results summary
├── static/
│   ├── css/style.css
│   └── images/bg.png
└── templates/
    └── index.html            # Prediction form UI
```

## How to Run

```bash
pip install flask pandas numpy scikit-learn
python app.py
```
Then open `http://127.0.0.1:5000` in your browser.
##  Screenshot

<img width="1897" height="973" alt="Screenshot 2026-10-01 133147" src="https://github.com/user-attachments/assets/9a6722d3-408a-4f43-9fc2-93ef25419b87" />

