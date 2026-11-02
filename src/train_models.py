"""
Real Estate Investment Advisor: Model Training & MLflow Tracking Module
Trains, evaluates, and logs both Classification ("Good Investment")
and Regression ("Future Price in 5 Years") models.
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")

MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Feature definitions
NUMERIC_FEATURES = [
    'BHK',
    'Size_in_SqFt',
    'Price_in_Lakhs',
    'Price_per_SqFt_INR',
    'Age_of_Property',
    'Floor_No',
    'Total_Floors',
    'Nearby_Schools',
    'Nearby_Hospitals',
    'Amenities_Count'
]

CATEGORICAL_FEATURES = [
    'State',
    'City',
    'Property_Type',
    'Furnished_Status',
    'Public_Transport_Accessibility',
    'Parking_Space',
    'Security',
    'Facing',
    'Owner_Type',
    'Availability_Status'
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def get_preprocessor():
    """Builds reusable preprocessor for numerical and categorical features."""
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, NUMERIC_FEATURES),
            ('cat', cat_pipeline, CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor


def train_and_track_models(data_path="data/processed/cleaned_housing_data.csv"):
    print("=" * 70)
    print("REAL ESTATE INVESTMENT ADVISOR: MODEL TRAINING & MLFLOW TRACKING")
    print("=" * 70)

    # Configure MLflow with SQLite backend
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    experiment_name = "Real_Estate_Investment_Advisor"
    mlflow.set_experiment(experiment_name)
    print(f"MLflow tracking initialized for experiment: '{experiment_name}'")

    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded processed dataset with {len(df):,} rows.")

    # Subsample 75,000 rows for high-fidelity, high-speed training and evaluation
    if len(df) > 75000:
        sample_df = df.sample(n=75000, random_state=42)
    else:
        sample_df = df.copy()

    X = sample_df[ALL_FEATURES].copy()
    y_class = sample_df['Good_Investment'].astype(int)
    y_reg = sample_df['Future_Price_5Y'].astype(float)

    # Train / Test Split
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    evaluation_records = []

    # -------------------------------------------------------------
    # 1. CLASSIFICATION MODELS (Target: Good_Investment)
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("TRAINING CLASSIFICATION MODELS (Target: Good_Investment)")
    print("=" * 50)

    classifiers = {
        "Random_Forest_Classifier": RandomForestClassifier(
            n_estimators=150, max_depth=15, random_state=42, n_jobs=-1, class_weight="balanced"
        ),
        "Hist_Gradient_Boosting_Classifier": HistGradientBoostingClassifier(
            max_iter=150, max_depth=12, random_state=42
        ),
        "Logistic_Regression": LogisticRegression(
            max_iter=500, random_state=42
        )
    }

    best_classifier = None
    best_clf_f1 = -1.0
    best_clf_name = ""

    for name, clf_model in classifiers.items():
        print(f"\n--- Training {name} ---")
        pipeline = Pipeline([
            ('preprocessor', get_preprocessor()),
            ('classifier', clf_model)
        ])

        with mlflow.start_run(run_name=f"Classification_{name}"):
            pipeline.fit(X_train_c, y_train_c)
            y_pred = pipeline.predict(X_test_c)
            y_prob = pipeline.predict_proba(X_test_c)[:, 1] if hasattr(pipeline, "predict_proba") else y_pred

            acc = accuracy_score(y_test_c, y_pred)
            prec = precision_score(y_test_c, y_pred)
            rec = recall_score(y_test_c, y_pred)
            f1 = f1_score(y_test_c, y_pred)
            roc = roc_auc_score(y_test_c, y_prob)

            print(f"Accuracy:  {acc:.4f}")
            print(f"Precision: {prec:.4f}")
            print(f"Recall:    {rec:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print(f"ROC-AUC:   {roc:.4f}")

            # Log to MLflow
            mlflow.log_param("model_name", name)
            mlflow.log_param("task", "classification")
            mlflow.log_params(clf_model.get_params())
            mlflow.log_metrics({
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "roc_auc": roc
            })
            mlflow.sklearn.log_model(pipeline, name="model", serialization_format="pickle")

            evaluation_records.append({
                "Task": "Classification",
                "Model": name,
                "Metric_1_Name": "Accuracy",
                "Metric_1_Value": round(acc, 4),
                "Metric_2_Name": "F1_Score",
                "Metric_2_Value": round(f1, 4),
                "Metric_3_Name": "ROC_AUC",
                "Metric_3_Value": round(roc, 4)
            })

            if f1 > best_clf_f1:
                best_clf_f1 = f1
                best_classifier = pipeline
                best_clf_name = name

    print(f"\nSelected Best Classifier: {best_clf_name} (F1: {best_clf_f1:.4f})")
    joblib.dump(best_classifier, os.path.join(MODELS_DIR, "investment_classifier.pkl"))

    # Generate Confusion Matrix & ROC Curve for Best Classifier
    y_pred_best = best_classifier.predict(X_test_c)
    y_prob_best = best_classifier.predict_proba(X_test_c)[:, 1]

    # Confusion matrix plot
    cm = confusion_matrix(y_test_c, y_pred_best)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Good (0)', 'Good (1)'],
                yticklabels=['Not Good (0)', 'Good (1)'])
    plt.title(f"Confusion Matrix - {best_clf_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # ROC Curve plot
    fpr, tpr, _ = roc_curve(y_test_c, y_prob_best)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='#1f77b4', lw=2, label=f"ROC Curve (AUC = {roc_auc_score(y_test_c, y_prob_best):.3f})")
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.title(f"ROC Curve - {best_clf_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(OUTPUTS_DIR, "roc_curve.png")
    plt.savefig(roc_path, dpi=200)
    plt.close()

    # Feature Importance for Classifier (if RandomForest)
    if "Random_Forest" in best_clf_name:
        rf = best_classifier.named_steps['classifier']
        prep = best_classifier.named_steps['preprocessor']
        ohe = prep.named_transformers_['cat'].named_steps['encoder']
        cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
        all_feature_names = NUMERIC_FEATURES + cat_names

        importances = pd.Series(rf.feature_importances_, index=all_feature_names).sort_values(ascending=False).head(15)
        plt.figure(figsize=(10, 6))
        importances.plot(kind='barh', color='#2ca02c')
        plt.title("Top 15 Feature Importances - Investment Classifier")
        plt.xlabel("Relative Importance")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_DIR, "feature_importance_classifier.png"), dpi=200)
        plt.close()

    # -------------------------------------------------------------
    # 2. REGRESSION MODELS (Target: Future_Price_5Y)
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("TRAINING REGRESSION MODELS (Target: Future_Price_5Y)")
    print("=" * 50)

    regressors = {
        "Random_Forest_Regressor": RandomForestRegressor(
            n_estimators=150, max_depth=16, random_state=42, n_jobs=-1
        ),
        "Hist_Gradient_Boosting_Regressor": HistGradientBoostingRegressor(
            max_iter=150, max_depth=12, random_state=42
        ),
        "Ridge_Regression": Ridge(alpha=1.0)
    }

    best_regressor = None
    best_reg_r2 = -float("inf")
    best_reg_name = ""

    for name, reg_model in regressors.items():
        print(f"\n--- Training {name} ---")
        pipeline = Pipeline([
            ('preprocessor', get_preprocessor()),
            ('regressor', reg_model)
        ])

        with mlflow.start_run(run_name=f"Regression_{name}"):
            pipeline.fit(X_train_r, y_train_r)
            y_pred_r = pipeline.predict(X_test_r)

            mae = mean_absolute_error(y_test_r, y_pred_r)
            mse = mean_squared_error(y_test_r, y_pred_r)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_r, y_pred_r)

            print(f"MAE:      {mae:.4f}")
            print(f"MSE:      {mse:.4f}")
            print(f"RMSE:     {rmse:.4f}")
            print(f"R2-Score: {r2:.4f}")

            # Log to MLflow
            mlflow.log_param("model_name", name)
            mlflow.log_param("task", "regression")
            mlflow.log_params(reg_model.get_params())
            mlflow.log_metrics({
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "r2_score": r2
            })
            mlflow.sklearn.log_model(pipeline, name="model", serialization_format="pickle")

            evaluation_records.append({
                "Task": "Regression",
                "Model": name,
                "Metric_1_Name": "MAE",
                "Metric_1_Value": round(mae, 4),
                "Metric_2_Name": "RMSE",
                "Metric_2_Value": round(rmse, 4),
                "Metric_3_Name": "R2_Score",
                "Metric_3_Value": round(r2, 4)
            })

            if r2 > best_reg_r2:
                best_reg_r2 = r2
                best_regressor = pipeline
                best_reg_name = name

    print(f"\nSelected Best Regressor: {best_reg_name} (R2: {best_reg_r2:.4f})")
    joblib.dump(best_regressor, os.path.join(MODELS_DIR, "future_price_regressor.pkl"))

    # Generate Actual vs Predicted & Residuals for Best Regressor
    y_pred_r_best = best_regressor.predict(X_test_r)
    residuals = y_test_r - y_pred_r_best

    # Actual vs Predicted plot
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test_r[:1500], y_pred_r_best[:1500], alpha=0.4, color='#1f77b4', edgecolor='none', s=20)
    plt.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r--', lw=2)
    plt.title(f"Actual vs. Predicted 5Y Price - {best_reg_name}")
    plt.xlabel("Actual 5-Year Price (Lakhs)")
    plt.ylabel("Predicted 5-Year Price (Lakhs)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "regression_actual_vs_predicted.png"), dpi=200)
    plt.close()

    # Residuals plot
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, bins=40, color='#d62728')
    plt.title(f"Residuals Distribution - {best_reg_name}")
    plt.xlabel("Residual (Actual - Predicted in Lakhs)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "residuals_plot.png"), dpi=200)
    plt.close()

    # Feature Importance for Regressor (if RandomForest)
    if "Random_Forest" in best_reg_name:
        rf_reg = best_regressor.named_steps['regressor']
        prep = best_regressor.named_steps['preprocessor']
        ohe = prep.named_transformers_['cat'].named_steps['encoder']
        cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
        all_feature_names = NUMERIC_FEATURES + cat_names

        importances = pd.Series(rf_reg.feature_importances_, index=all_feature_names).sort_values(ascending=False).head(15)
        plt.figure(figsize=(10, 6))
        importances.plot(kind='barh', color='#ff7f0e')
        plt.title("Top 15 Feature Importances - Future Price Regressor")
        plt.xlabel("Relative Importance")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_DIR, "feature_importance_regressor.png"), dpi=200)
        plt.close()

    # Save Model Evaluation Summary
    eval_df = pd.DataFrame(evaluation_records)
    eval_df.to_csv(os.path.join(OUTPUTS_DIR, "model_evaluation.csv"), index=False)
    print("\nSaved evaluation summary table to outputs/model_evaluation.csv")

    # Save Model Metadata
    metadata = {
        "best_classifier": best_clf_name,
        "best_classifier_f1": round(best_clf_f1, 4),
        "best_regressor": best_reg_name,
        "best_regressor_r2": round(best_reg_r2, 4),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "all_features": ALL_FEATURES,
        "cities": sorted(df['City'].unique().tolist()),
        "states": sorted(df['State'].unique().tolist()),
        "property_types": sorted(df['Property_Type'].unique().tolist()),
        "furnished_statuses": sorted(df['Furnished_Status'].unique().tolist()),
        "facings": sorted(df['Facing'].unique().tolist()),
        "owner_types": sorted(df['Owner_Type'].unique().tolist()),
        "availability_statuses": sorted(df['Availability_Status'].unique().tolist()),
        "transport_options": ['High', 'Medium', 'Low']
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print("=" * 70)
    print("TRAINING AND MLFLOW LOGGING COMPLETED SUCCESSFULLY!")
    print(f"Models saved in '{MODELS_DIR}/'")
    print(f"Artifacts saved in '{OUTPUTS_DIR}/'")
    print("=" * 70)


if __name__ == "__main__":
    train_and_track_models()
