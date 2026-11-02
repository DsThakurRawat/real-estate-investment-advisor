"""
Comprehensive verification test suite for the Real Estate Investment Advisor FastAPI application.
"""

import json
import urllib.request
import urllib.parse
import sys

BASE_URL = "http://127.0.0.1:8000"

def test_endpoint(name, url, method="GET", data=None):
    print(f"\n[TEST] {name} ({method} {url})")
    try:
        req = urllib.request.Request(
            url,
            method=method,
            headers={"Content-Type": "application/json"}
        )
        if data:
            req.data = json.dumps(data).encode("utf-8")
        
        with urllib.request.urlopen(req) as resp:
            status = resp.status
            content_type = resp.headers.get("content-type", "")
            body = resp.read()
            print(f"  Status: {status} | Content-Type: {content_type}")
            if "application/json" in content_type:
                json_data = json.loads(body.decode("utf-8"))
                print(f"  Response summary: {str(json_data)[:150]}...")
                return json_data
            else:
                print(f"  Response size: {len(body)} bytes")
                return body
    except Exception as e:
        print(f"  FAILED: {e}")
        return None

def run_all_tests():
    print("=" * 60)
    print("RUNNING COMPREHENSIVE RE-VERIFICATION SUITE")
    print("=" * 60)

    # 1. Health check
    h = test_endpoint("Health Check", f"{BASE_URL}/health")
    assert h is not None and h["status"] == "healthy", "Health check failed"

    # 2. Metadata
    meta = test_endpoint("Market Metadata", f"{BASE_URL}/market/metadata")
    assert meta is not None and "cities" in meta, "Metadata failed"
    print(f"  Supported Cities: {len(meta['cities'])}, States: {len(meta['states'])}")

    # 3. Market Properties (Filter)
    props = test_endpoint("Filtered Properties Search", f"{BASE_URL}/market/properties?city=Pune&limit=3")
    assert props is not None and len(props) > 0, "Property search failed"
    print(f"  Retrieved {len(props)} sample properties for Pune.")

    # 4. Predict Endpoint - Property 1 (High Potential)
    prop_good = {
        "State": "Maharashtra",
        "City": "Pune",
        "Property_Type": "Apartment",
        "BHK": 3,
        "Size_in_SqFt": 2000.0,
        "Price_in_Lakhs": 95.0,  # Below median price/sqft
        "Age_of_Property": 3,
        "Furnished_Status": "Furnished",
        "Floor_No": 7,
        "Total_Floors": 18,
        "Nearby_Schools": 9,
        "Nearby_Hospitals": 8,
        "Public_Transport_Accessibility": "High",
        "Parking_Space": "Yes",
        "Security": "Yes",
        "Facing": "East",
        "Owner_Type": "Owner",
        "Availability_Status": "Ready_to_Move",
        "Amenities": ["Gym", "Pool", "Clubhouse", "Garden", "Playground"]
    }
    pred_res_good = test_endpoint("Prediction - High Potential Listing", f"{BASE_URL}/predict", method="POST", data=prop_good)
    assert pred_res_good is not None, "Predict good failed"
    print(f"  Verdict: {pred_res_good['verdict']} (Confidence: {pred_res_good['confidence_score_pct']}%)")
    print(f"  Current Price: ₹{pred_res_good['current_price_lakhs']}L -> 5Y Price: ₹{pred_res_good['estimated_future_price_5y_lakhs']}L (ROI: +{pred_res_good['expected_5y_roi_pct']}%)")

    # 5. Predict Endpoint - Property 2 (Overpriced / Sub-optimal)
    prop_bad = {
        "State": "Maharashtra",
        "City": "Pune",
        "Property_Type": "Apartment",
        "BHK": 1,
        "Size_in_SqFt": 600.0,
        "Price_in_Lakhs": 150.0,  # Highly inflated price/sqft
        "Age_of_Property": 25,
        "Furnished_Status": "Unfurnished",
        "Floor_No": 1,
        "Total_Floors": 4,
        "Nearby_Schools": 2,
        "Nearby_Hospitals": 2,
        "Public_Transport_Accessibility": "Low",
        "Parking_Space": "No",
        "Security": "No",
        "Facing": "South",
        "Owner_Type": "Broker",
        "Availability_Status": "Under_Construction",
        "Amenities": []
    }
    pred_res_bad = test_endpoint("Prediction - Overpriced Listing", f"{BASE_URL}/predict", method="POST", data=prop_bad)
    assert pred_res_bad is not None, "Predict bad failed"
    print(f"  Verdict: {pred_res_bad['verdict']} (Confidence: {pred_res_bad['confidence_score_pct']}%)")
    print(f"  Current Price: ₹{pred_res_bad['current_price_lakhs']}L -> 5Y Price: ₹{pred_res_bad['estimated_future_price_5y_lakhs']}L")

    # 6. Batch Predict
    batch_req = {"properties": [prop_good, prop_bad]}
    batch_res = test_endpoint("Batch Prediction", f"{BASE_URL}/predict/batch", method="POST", data=batch_req)
    assert batch_res is not None and len(batch_res) == 2, "Batch predict failed"

    # 7. Model Leaderboard
    lb = test_endpoint("Model Leaderboard", f"{BASE_URL}/models/leaderboard")
    assert lb is not None and len(lb) > 0, "Leaderboard failed"

    # 8. EDA Insights
    eda_data = test_endpoint("EDA Insights", f"{BASE_URL}/eda/insights")
    assert eda_data is not None, "EDA insights failed"

    # 9. Dashboard UI
    dash = test_endpoint("FastAPI Dashboard UI", f"{BASE_URL}/")
    assert dash is not None and b"Real Estate Investment Advisor" in dash, "Dashboard failed"

    # 10. Swagger Docs
    docs = test_endpoint("Swagger OpenAPI Docs", f"{BASE_URL}/docs")
    assert docs is not None and b"Swagger UI" in docs, "Docs failed"

    # 11. Static EDA Visual Asset
    img = test_endpoint("Static Visual Asset (01_price_distribution.png)", f"{BASE_URL}/static/outputs/eda_visualizations/01_price_distribution.png")
    assert img is not None and len(img) > 1000, "Static image failed"

    # 12. Static Confusion Matrix
    cm_img = test_endpoint("Static Confusion Matrix (confusion_matrix.png)", f"{BASE_URL}/static/outputs/confusion_matrix.png")
    assert cm_img is not None and len(cm_img) > 1000, "Confusion matrix image failed"

    print("\n" + "=" * 60)
    print("ALL 11 VERIFICATION TESTS PASSED SUCCESSFULLY! (100% HEALTHY)")
    print("=" * 60)

if __name__ == "__main__":
    run_all_tests()
