#  Real Estate Investment Advisor: Predicting Property Profitability & Future Value

An end-to-end Machine Learning, Exploratory Data Analysis, and Decision-Support platform powered by **FastAPI**, **Scikit-Learn**, and **MLflow**, designed to empower real estate investors, property buyers, and institutional analysts in evaluating investment opportunities across 42 Indian cities.

---

##  Executive Summary & Problem Statement

Real estate purchases represent significant capital commitments with multi-year holding periods. Investors and retail buyers struggle with:
1. **Investment Viability:** Objectively classifying whether a property listing is a high-conviction "Good Investment" versus an overpriced, high-risk asset.
2. **Future Valuation Forecasting:** Accurately projecting the estimated property price after 5 years based on macro location tiers, infrastructure, connectivity, and asset quality.

This project implements:
- **Binary Classification Model (`Good_Investment`):** Predicts investment viability (`1 = Good Investment`, `0 = Sub-optimal / High Risk`) with **99.2% Accuracy** and **0.9994 ROC-AUC**.
- **Continuous Regression Model (`Future_Price_5Y`):** Forecasts estimated property price in 5 years with an **R^2 of 0.9999** and **MAE of 1.84 Lakhs INR**.
- **FastAPI Enterprise Backend:** Asynchronous REST API with Pydantic validation, OpenAPI / Swagger documentation (`/docs`), and low-latency inference.
- **FastAPI Modern Web Dashboard (`/`):** Responsive single-page application with real-time prediction forms, 5-year growth trajectory plots, radar dimension charts, 250k property explorer, and mortgage cashflow simulator.
- **MLflow Experiment Tracking:** Version-controlled models, metric logs, hyperparameter audits, and diagnostic visual artifacts in `mlflow.db`.

---

##  Project Repository Structure

```
assignment/
|-- data/
|   |-- raw/
|   |   \-- india_housing_prices.csv        # 250,000 raw property records
|   \-- processed/
|       \-- cleaned_housing_data.csv        # Cleaned dataset with 36 engineered features
|-- models/
|   |-- investment_classifier.pkl           # Production Classifier (HistGradientBoosting)
|   |-- future_price_regressor.pkl          # Production Regressor (HistGradientBoosting)
|   \-- model_metadata.json                 # Feature lists, target categories & metadata
|-- outputs/
|   |-- eda_visualizations/                 # 20 publication-grade EDA figures
|   |   |-- 01_price_distribution.png
|   |   |-- 02_size_distribution.png
|   |   |-- ...
|   |   \-- 20_transport_vs_investment.png
|   |-- confusion_matrix.png                # Best model confusion matrix
|   |-- roc_curve.png                       # Best model ROC-AUC curve
|   |-- regression_actual_vs_predicted.png  # Parity plot (Actual vs 5Y Predicted)
|   |-- residuals_plot.png                  # Error distribution
|   |-- model_evaluation.csv                # Leaderboard metrics table
|   \-- eda_insights.json                   # Numerical summary of EDA questions
|-- src/
|   |-- __init__.py
|   |-- preprocessing.py                    # Cleaning, targets engineering, domain logic
|   |-- eda.py                              # Script answering all 20 project EDA questions
|   \-- train_models.py                     # ML pipeline with MLflow tracking & evaluation
|-- templates/
|   \-- index.html                          # Modern glassmorphism SPA dashboard for FastAPI
|-- main.py                                 # FastAPI enterprise application & API routes
|-- verify_all.py                           # 12-point automated test suite for endpoints
|-- requirements.txt                        # Environment dependencies
|-- mlflow.db                               # SQLite backend database for MLflow
\-- README.md                               # Comprehensive documentation
```

---

##  FastAPI REST Endpoints

The application is deployed on `http://127.0.0.1:8000`:

| HTTP Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Interactive Web Dashboard (Single-Page Application). |
| `GET` | `/docs` | Interactive Swagger / OpenAPI documentation & testing sandbox. |
| `GET` | `/health` | System health check, model readiness, and dataset statistics. |
| `POST` | `/predict` | Evaluates property investment viability & 5-year forecasted price. |
| `POST` | `/predict/batch` | Bulk valuation and investment classification on multiple listings. |
| `GET` | `/market/metadata` | Supported cities (42), states (20), property types, and mappings. |
| `GET` | `/market/properties` | Paginated search and filtering over 250,000 properties. |
| `GET` | `/eda/insights` | Statistical answers and JSON metrics for all 20 EDA questions. |
| `GET` | `/models/leaderboard`| Model performance comparison table (Accuracy, F1, MAE, R^2). |
| `GET` | `/static/outputs/*` | Serves all 20 EDA charts and model diagnostic artifacts. |

### Sample Inference Payload (`POST /predict`)
```json
{
  "State": "Maharashtra",
  "City": "Pune",
  "Property_Type": "Apartment",
  "BHK": 3,
  "Size_in_SqFt": 1850.0,
  "Price_in_Lakhs": 125.0,
  "Age_of_Property": 4,
  "Furnished_Status": "Semi-furnished",
  "Floor_No": 5,
  "Total_Floors": 15,
  "Nearby_Schools": 8,
  "Nearby_Hospitals": 7,
  "Public_Transport_Accessibility": "High",
  "Parking_Space": "Yes",
  "Security": "Yes",
  "Facing": "East",
  "Owner_Type": "Owner",
  "Availability_Status": "Ready_to_Move",
  "Amenities": ["Gym", "Pool", "Clubhouse", "Garden"]
}
```

### Sample Prediction Response
```json
{
  "is_good_investment": true,
  "good_investment_code": 1,
  "confidence_score_pct": 100.0,
  "verdict": " RECOMMENDED INVESTMENT",
  "current_price_lakhs": 125.0,
  "price_per_sqft_inr": 6756.76,
  "estimated_future_price_5y_lakhs": 206.11,
  "projected_capital_gain_lakhs": 81.11,
  "expected_5y_roi_pct": 64.89,
  "cagr_pct": 10.52,
  "city_median_price_per_sqft_inr": 9256.45,
  "price_diff_vs_city_median_pct": -27.0,
  "investment_health_dimensions": {
    "Pricing Advantage": 100,
    "Infrastructure": 75,
    "Connectivity": 100,
    "Amenities": 80,
    "Possession Safety": 95
  }
}
```

---

##  Model Training & MLflow Leaderboard

All model experiments were tracked using MLflow with SQLite backend (`sqlite:///mlflow.db`).

### Classification Performance (`Good_Investment`)
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
|---|---|---|---|---|---|---|
| **HistGradientBoostingClassifier** | **99.21%** | **98.95%** | **99.49%** | **0.9922** | **0.9994** | [Selected] **Production Selected** |
| RandomForestClassifier | 92.66% | 91.09% | 94.75% | 0.9289 | 0.9857 | Candidate |
| LogisticRegression (Baseline) | 84.10% | 84.53% | 83.92% | 0.8423 | 0.9243 | Baseline |

### Regression Performance (`Future_Price_5Y`)
| Model | MAE (Lakhs) | MSE | RMSE (Lakhs) | R^2 Score | Status |
|---|---|---|---|---|---|
| **HistGradientBoostingRegressor** | **1.84 Lakhs** | **6.16** | **2.48 Lakhs** | **0.9999** | [Selected] **Production Selected** |
| RandomForestRegressor | 9.60 Lakhs | 163.83 | 12.80 Lakhs | 0.9964 | Candidate |
| RidgeRegression (Baseline) | 7.80 Lakhs | 109.07 | 10.44 Lakhs | 0.9976 | Baseline |

---

##  Comprehensive 20-Point EDA Findings

| Section | Question | Key Statistical Finding | Visual Artifact |
|---|---|---|---|
| **Price & Size** | **Q1: Price Distribution** | Prices range from INR 10L to INR 500L; mean is INR 254.59L, median is INR 254.50L | `01_price_distribution.png` |
| **Price & Size** | **Q2: Size Distribution** | Areas span 500 to 5,000 SqFt; mean is 2,750 SqFt across 1-5 BHK configurations | `02_size_distribution.png` |
| **Price & Size** | **Q3: Price/SqFt by Type** | Villas command premium per sqft over Independent Houses and Apartments | `03_price_per_sqft_by_type.png` |
| **Price & Size** | **Q4: Size vs. Price** | Strong positive correlation with distinct pricing tiers by property type | `04_size_vs_price.png` |
| **Price & Size** | **Q5: Outlier Analysis** | Low incidence of anomalous extremes; well-behaved IQR bounds | `05_outliers_price_size.png` |
| **Location** | **Q6: Avg Price/SqFt by State** | Western & Southern states (Maharashtra, Tamil Nadu, Karnataka) lead averages | `06_avg_price_per_sqft_by_state.png` |
| **Location** | **Q7: Avg Price by City** | New Delhi, Mumbai, Pune, and Chennai lead in average transaction ticket size | `07_avg_price_by_city.png` |
| **Location** | **Q8: Median Age by Locality** | Central localities average 25-35 years; developing peripheries average <10 years | `08_median_age_by_locality.png` |
| **Location** | **Q9: BHK Distribution by City** | 2 BHK and 3 BHK units constitute over 60% of all metro inventory | `09_bhk_distribution_across_cities.png` |
| **Location** | **Q10: Top 5 Expensive Localities** | Prime localities reflect 25-35% price premiums over respective city medians | `10_top5_expensive_localities.png` |
| **Relationships** | **Q11: Feature Correlation** | High correlation between Current Price and 5Y Valuation ($r > 0.99$) | `11_correlation_heatmap.png` |
| **Relationships** | **Q12: Schools vs. Price/SqFt** | School density acts as a critical anchor for sustained neighborhood appreciation | `12_schools_vs_price_per_sqft.png` |
| **Relationships** | **Q13: Hospitals vs. Price/SqFt** | Proximity to healthcare centers correlates with premium real estate pricing | `13_hospitals_vs_price_per_sqft.png` |
| **Relationships** | **Q14: Furnished Status** | Furnished homes trade at 8-12% higher ticket values compared to bare shell | `14_price_by_furnished_status.png` |
| **Relationships** | **Q15: Facing Direction** | East and North facing properties command modest buyer liquidity premiums | `15_price_by_facing.png` |
| **Ownership** | **Q16: Owner Type Share** | Balanced inventory split across Owner resales, Builder stock, and Broker listings | `16_owner_type_distribution.png` |
| **Ownership** | **Q17: Availability Status** | 50.01% Ready-to-Move vs. 49.99% Under-Construction listings | `17_availability_status.png` |
| **Ownership** | **Q18: Parking Space Impact** | Properties with dedicated parking trade higher and offer better resale velocity | `18_parking_vs_price.png` |
| **Ownership** | **Q19: Amenities Impact** | Communities with 4+ amenities command superior price realization per sqft | `19_amenities_vs_price_per_sqft.png` |
| **Ownership** | **Q20: Public Transport Impact** | High transit accessibility boosts 'Good Investment' probability by over 35% | `20_transport_vs_investment.png` |

---

##  Quickstart & Verification Guide

### 1. Prerequisites & Environment Setup
```bash
# Navigate to the workspace
cd /home/divyansh-rawat/assignment

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Preprocess Data & Train Models
```bash
# Preprocessing and target engineering
python src/preprocessing.py

# Generate all 20 EDA visual charts
python src/eda.py

# Train models and log to MLflow
python src/train_models.py
```

### 3. Launch FastAPI Server
```bash
# Starts Uvicorn server on port 8000
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

- **Web Dashboard:** Visit `http://127.0.0.1:8000/`
- **Swagger Documentation:** Visit `http://127.0.0.1:8000/docs`

### 4. Run Automated Test & Re-verification Suite
```bash
python verify_all.py
```
This tests and validates all 12 API endpoints, inference contracts, and static file deliveries.
