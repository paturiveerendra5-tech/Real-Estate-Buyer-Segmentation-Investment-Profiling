# ============================================================
# FEATURE ENGINEERING MODULE
# Real Estate Buyer Segmentation & Investment Profiling
# ============================================================

import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.preprocessing import StandardScaler


# Step 1: Project Paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models"

PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
MODEL_PATH.mkdir(parents=True, exist_ok=True)


# Step 2: Select Features

def select_features(clients):

    selected_columns = [
        "client_id",
        "client_type",
        "gender",
        "country",
        "region",
        "acquisition_purpose",
        "loan_applied",
        "satisfaction_score"
    ]

    missing_columns = [
        column for column in selected_columns
        if column not in clients.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    features = clients[selected_columns].copy()

    if features.empty:
        raise ValueError("Client dataset is empty!")

    if features["client_id"].isnull().any():
        raise ValueError("Missing client IDs found!")

    if features["client_id"].duplicated().any():
        raise ValueError("Duplicate client IDs found!")

    return features


# Step 3: Handle Missing Values

def handle_missing_values(features):

    features = features.copy()

    categorical_columns = [
        "client_type",
        "gender",
        "country",
        "region",
        "acquisition_purpose",
        "loan_applied"
    ]

    for column in categorical_columns:

        features[column] = (
            features[column]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
            .fillna("Unknown")
        )

    features["satisfaction_score"] = pd.to_numeric(
        features["satisfaction_score"],
        errors="coerce"
    )

    median_value = features["satisfaction_score"].median()

    if pd.isna(median_value):
        median_value = 0

    features["satisfaction_score"] = (
        features["satisfaction_score"].fillna(median_value)
    )

    return features


# Step 4: Separate Client IDs

def separate_client_ids(features):

    client_ids = features[["client_id"]].copy()

    model_features = features.drop(
        columns=["client_id"]
    ).copy()

    return model_features, client_ids


# Step 5: Encode Categorical Features

def encode_features(features):

    categorical_columns = [
        "client_type",
        "gender",
        "country",
        "region",
        "acquisition_purpose",
        "loan_applied"
    ]

    encoded_features = pd.get_dummies(
        features,
        columns=categorical_columns,
        dtype=int
    )

    encoded_features = encoded_features.astype(float)

    print("\nCategorical Encoding Completed!")

    print("Encoded Features Shape:", encoded_features.shape)

    return encoded_features


# Step 6: Scale Numerical Features

def scale_features(encoded_features):

    encoded_features = encoded_features.copy()

    if encoded_features.empty:
        raise ValueError("No features available for scaling!")

    scaler = StandardScaler()

    scaled_array = scaler.fit_transform(encoded_features)

    scaled_features = pd.DataFrame(
        scaled_array,
        columns=encoded_features.columns,
        index=encoded_features.index
    )

    print("\nFeature Scaling Completed!")

    return scaled_features, scaler


# Step 7: Save Feature Engineering Results

def save_feature_results(
    scaled_features,
    client_ids,
    scaler
):

    # Save Scaled Features

    scaled_features.to_csv(
        PROCESSED_PATH / "scaled_features.csv",
        index=False
    )

    # Save Client IDs Separately

    client_ids.to_csv(
        PROCESSED_PATH / "client_ids.csv",
        index=False
    )

    # Save Encoded Feature Names

    feature_names = pd.DataFrame({
        "feature_name": scaled_features.columns
    })

    feature_names.to_csv(
        PROCESSED_PATH / "feature_names.csv",
        index=False
    )

    # Save Scaler

    joblib.dump(
        scaler,
        MODEL_PATH / "standard_scaler.pkl"
    )

    print("\nFeature Engineering Files Saved Successfully!")


# Step 8: Main Feature Engineering Pipeline

def run_feature_engineering(clients):

    print("\n====================================")
    print(" REAL ESTATE FEATURE ENGINEERING")
    print("====================================")

    # Select Features

    features = select_features(clients)

    # Handle Missing Values

    features = handle_missing_values(features)

    # Separate IDs

    model_features, client_ids = separate_client_ids(
        features
    )

    # Encode Features

    encoded_features = encode_features(model_features)

    # Scale Features

    scaled_features, scaler = scale_features(
        encoded_features
    )

    # Save Results

    save_feature_results(
        scaled_features,
        client_ids,
        scaler
    )

    print("\nFEATURE ENGINEERING SUMMARY")
    print("----------------------------")

    print("Original Records:", len(clients))

    print("Client IDs Shape:", client_ids.shape)

    print("Encoded Features Shape:", encoded_features.shape)

    print("Scaled Features Shape:", scaled_features.shape)

    print("\n====================================")
    print(" FEATURE ENGINEERING COMPLETED!")
    print("====================================")

    return scaled_features, client_ids, scaler


# Step 9: Execute Main Function

if __name__ == "__main__":

    cleaned_file = PROCESSED_PATH / "cleaned_clients.csv"

    if not cleaned_file.exists():
        raise FileNotFoundError(
            "Cleaned clients dataset not found. "
            "Run data_cleaning.py first."
        )

    clients = pd.read_csv(cleaned_file)

    run_feature_engineering(clients)