"""
Real Estate Investment Advisor: FastAPI Enterprise Backend & Application
Serves ML inference for investment classification and 5-year valuation forecasting,
market analytics, and an interactive dashboard.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# GLOBAL RESOURCES & LIFESPAN
# -----------------------------------------------------------------------------
DATA_PATH = "data/processed/cleaned_housing_data.csv"
CLASSIFIER_PATH = "models/investment_classifier.pkl"
REGRESSOR_PATH = "models/future_price_regressor.pkl"
METADATA_PATH = "models/model_metadata.json"
OUTPUTS_DIR = "outputs"

resources: Dict[str, Any] = {
    "df": None,
    "classifier": None,
    "regressor": None,
    "metadata": None,
    "city_medians": None,
    "state_city_map": None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("STARTING REAL ESTATE INVESTMENT ADVISOR FASTAPI SERVER")
    print("=" * 60)

    # 1. Load Data
    if os.path.exists(DATA_PATH):
        print(f"Loading processed dataset from {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH)
        resources["df"] = df
        resources["city_medians"] = df.groupby('City')['Price_per_SqFt_INR'].median().to_dict()
        
        # Build state-city mapping
        state_city_map = {}
        for state, group in df.groupby('State'):
            state_city_map[state] = sorted(group['City'].unique().tolist())
        resources["state_city_map"] = state_city_map
        print(f"Dataset ready with {len(df):,} records across {len(resources['city_medians'])} cities.")
    else:
        print(f"WARNING: Dataset not found at {DATA_PATH}")

    # 2. Load Models
    if os.path.exists(CLASSIFIER_PATH):
        print(f"Loading classifier from {CLASSIFIER_PATH}...")
        resources["classifier"] = joblib.load(CLASSIFIER_PATH)
    if os.path.exists(REGRESSOR_PATH):
        print(f"Loading regressor from {REGRESSOR_PATH}...")
        resources["regressor"] = joblib.load(REGRESSOR_PATH)
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            resources["metadata"] = json.load(f)

    yield

    print("Shutting down Real Estate Investment Advisor API...")


app = FastAPI(
    title="Real Estate Investment Advisor API",
    description="Production Machine Learning API for Property Investment Classification & 5-Year Future Price Forecasting.",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for cross-origin integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount outputs directory as static files for visualizations
if os.path.exists(OUTPUTS_DIR):
    app.mount("/static/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")


# -----------------------------------------------------------------------------
# PYDANTIC DATA CONTRACTS
# -----------------------------------------------------------------------------
class PropertyInput(BaseModel):
    State: str = Field(..., example="Maharashtra")
    City: str = Field(..., example="Pune")
    Property_Type: str = Field(..., example="Apartment")
    BHK: int = Field(3, ge=1, le=5, example=3)
    Size_in_SqFt: float = Field(..., gt=100, example=1850.0)
    Price_in_Lakhs: float = Field(..., gt=1, example=125.0)
    Age_of_Property: int = Field(4, ge=0, le=100, example=4)
    Furnished_Status: str = Field("Semi-furnished", example="Semi-furnished")
    Floor_No: int = Field(5, ge=0, example=5)
    Total_Floors: int = Field(15, ge=1, example=15)
    Nearby_Schools: int = Field(8, ge=1, le=10, example=8)
    Nearby_Hospitals: int = Field(7, ge=1, le=10, example=7)
    Public_Transport_Accessibility: str = Field("High", example="High")
    Parking_Space: str = Field("Yes", example="Yes")
    Security: str = Field("Yes", example="Yes")
    Facing: str = Field("East", example="East")
    Owner_Type: str = Field("Owner", example="Owner")
    Availability_Status: str = Field("Ready_to_Move", example="Ready_to_Move")
    Amenities: Optional[List[str]] = Field(default_factory=lambda: ["Gym", "Pool", "Clubhouse", "Garden"])


class PredictionResult(BaseModel):
    is_good_investment: bool
    good_investment_code: int
    confidence_score_pct: float
    verdict: str
    current_price_lakhs: float
    price_per_sqft_inr: float
    estimated_future_price_5y_lakhs: float
    projected_capital_gain_lakhs: float
    expected_5y_roi_pct: float
    cagr_pct: float
    city_median_price_per_sqft_inr: float
    price_diff_vs_city_median_pct: float
    investment_health_dimensions: Dict[str, int]


class BatchPropertyInput(BaseModel):
    properties: List[PropertyInput]


# -----------------------------------------------------------------------------
# CORE API ENDPOINTS
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
async def serve_dashboard():
    """Serves the interactive web user interface."""
    template_path = "templates/index.html"
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Real Estate Investment Advisor API is running. Visit /docs for Swagger documentation.</h1>")


@app.get("/health", tags=["System"])
async def health_check():
    """Returns system status, model availability, and dataset stats."""
    return {
        "status": "healthy",
        "dataset_loaded": resources["df"] is not None,
        "dataset_rows": len(resources["df"]) if resources["df"] is not None else 0,
        "classifier_ready": resources["classifier"] is not None,
        "regressor_ready": resources["regressor"] is not None,
        "cities_supported": len(resources["city_medians"]) if resources["city_medians"] else 0,
        "version": "2.0.0"
    }


@app.get("/market/metadata", tags=["Market Data"])
async def get_market_metadata():
    """Returns dropdown options, supported cities, states, and mappings."""
    if not resources["metadata"]:
        raise HTTPException(status_code=503, detail="Model metadata is not loaded.")
    
    meta = resources["metadata"].copy()
    meta["state_city_map"] = resources["state_city_map"]
    return meta


@app.get("/market/properties", tags=["Market Data"])
async def get_properties(
    city: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    property_type: Optional[str] = Query(None),
    bhk: Optional[int] = Query(None),
    max_price: Optional[float] = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0)
):
    """Filters and returns property records from the 250,000 dataset."""
    df = resources["df"]
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded.")

    filtered = df
    if state:
        filtered = filtered[filtered['State'] == state]
    if city:
        filtered = filtered[filtered['City'] == city]
    if property_type:
        filtered = filtered[filtered['Property_Type'] == property_type]
    if bhk:
        filtered = filtered[filtered['BHK'] == bhk]
    if max_price:
        filtered = filtered[filtered['Price_in_Lakhs'] <= max_price]

    paginated = filtered.iloc[offset:offset + limit]
    cols = [
        'ID', 'State', 'City', 'Property_Type', 'BHK', 'Size_in_SqFt', 
        'Price_in_Lakhs', 'Price_per_SqFt_INR', 'Age_of_Property', 
        'Public_Transport_Accessibility', 'Good_Investment', 'Future_Price_5Y'
    ]
    return paginated[cols].to_dict(orient="records")


@app.post("/predict", response_model=PredictionResult, tags=["Predictions"])
async def predict_property(prop: PropertyInput):
    """
    Evaluates property investment viability and forecasts 5-year future price.
    """
    clf = resources["classifier"]
    reg = resources["regressor"]
    if clf is None or reg is None:
        raise HTTPException(status_code=503, detail="Models are not initialized.")

    # Calculate derived features
    psqft = round((prop.Price_in_Lakhs * 100000) / prop.Size_in_SqFt, 2)
    amenities_count = len(prop.Amenities) if prop.Amenities else 0

    input_df = pd.DataFrame([{
        'State': prop.State,
        'City': prop.City,
        'Property_Type': prop.Property_Type,
        'BHK': prop.BHK,
        'Size_in_SqFt': prop.Size_in_SqFt,
        'Price_in_Lakhs': prop.Price_in_Lakhs,
        'Price_per_SqFt_INR': psqft,
        'Age_of_Property': prop.Age_of_Property,
        'Furnished_Status': prop.Furnished_Status,
        'Floor_No': prop.Floor_No,
        'Total_Floors': prop.Total_Floors,
        'Nearby_Schools': prop.Nearby_Schools,
        'Nearby_Hospitals': prop.Nearby_Hospitals,
        'Public_Transport_Accessibility': prop.Public_Transport_Accessibility,
        'Parking_Space': prop.Parking_Space,
        'Security': prop.Security,
        'Amenities_Count': amenities_count,
        'Facing': prop.Facing,
        'Owner_Type': prop.Owner_Type,
        'Availability_Status': prop.Availability_Status
    }])

    # 1. Classification Inference
    clf_pred = int(clf.predict(input_df)[0])
    clf_prob = float(clf.predict_proba(input_df)[0][1]) if hasattr(clf, "predict_proba") else (0.95 if clf_pred == 1 else 0.05)

    # 2. Regression Inference
    reg_pred = float(reg.predict(input_df)[0])

    # 3. Financial Metrics
    capital_gain = round(reg_pred - prop.Price_in_Lakhs, 2)
    roi_pct = round((capital_gain / prop.Price_in_Lakhs) * 100, 2)
    cagr_pct = round((((reg_pred / prop.Price_in_Lakhs) ** (1 / 5)) - 1) * 100, 2)

    city_median_psqft = resources["city_medians"].get(prop.City, 9500.0) if resources["city_medians"] else 9500.0
    price_diff_pct = round(((psqft - city_median_psqft) / city_median_psqft) * 100, 2)

    # Health Dimensions (0-100)
    p_adv = max(10, min(100, int(100 - price_diff_pct)))
    infra_score = int(((prop.Nearby_Schools + prop.Nearby_Hospitals) / 20) * 100)
    conn_score = 100 if prop.Public_Transport_Accessibility == 'High' else (65 if prop.Public_Transport_Accessibility == 'Medium' else 30)
    amen_score = int((amenities_count / 5) * 100)
    safe_score = 95 if prop.Availability_Status == 'Ready_to_Move' else 55

    dims = {
        "Pricing Advantage": p_adv,
        "Infrastructure": infra_score,
        "Connectivity": conn_score,
        "Amenities": amen_score,
        "Possession Safety": safe_score
    }

    verdict = "🌟 RECOMMENDED INVESTMENT" if clf_pred == 1 else "⚠️ CAUTION / SUB-OPTIMAL"

    return PredictionResult(
        is_good_investment=bool(clf_pred == 1),
        good_investment_code=clf_pred,
        confidence_score_pct=round(clf_prob * 100, 1),
        verdict=verdict,
        current_price_lakhs=prop.Price_in_Lakhs,
        price_per_sqft_inr=psqft,
        estimated_future_price_5y_lakhs=round(reg_pred, 2),
        projected_capital_gain_lakhs=capital_gain,
        expected_5y_roi_pct=roi_pct,
        cagr_pct=cagr_pct,
        city_median_price_per_sqft_inr=city_median_psqft,
        price_diff_vs_city_median_pct=price_diff_pct,
        investment_health_dimensions=dims
    )


@app.post("/predict/batch", response_model=List[PredictionResult], tags=["Predictions"])
async def predict_batch(batch: BatchPropertyInput):
    """Performs bulk inference on a list of property specifications."""
    results = []
    for prop in batch.properties:
        res = await predict_property(prop)
        results.append(res)
    return results


@app.get("/eda/insights", tags=["EDA & Reports"])
async def get_eda_insights():
    """Returns numerical statistics and insights for all 20 EDA questions."""
    path = "outputs/eda_insights.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="EDA insights file not found.")


@app.get("/models/leaderboard", tags=["Model Registry"])
async def get_model_leaderboard():
    """Returns the MLflow model evaluation leaderboard across all architectures."""
    path = "outputs/model_evaluation.csv"
    if os.path.exists(path):
        df_eval = pd.read_csv(path)
        return df_eval.to_dict(orient="records")
    raise HTTPException(status_code=404, detail="Model evaluation table not found.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
