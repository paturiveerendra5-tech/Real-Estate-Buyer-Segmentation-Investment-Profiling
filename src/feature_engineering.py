# FEATURE ENGINEERING MODULE

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler


# Step 1: Select Features
def select_features(clients):

    features = clients.copy()

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

    features = features[selected_columns]

    return features


# Step 2: Handle Missing Values
def handle_missing_values(features):

    features = features.copy()

    for column in features.select_dtypes(include=["object", "category"]).columns:
        features[column] = features[column].fillna("Unknown")

    numeric_columns = features.select_dtypes(include=np.number).columns

    for column in numeric_columns:
        features[column] = features[column].fillna(
            features[column].median()
        )

    return features


# Step 3: Encode Categorical Features
def encode_features(features):

    features = features.copy()

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

    return encoded_features


# Step 4: Scale Features
def scale_features(encoded_features):

    encoded_features = encoded_features.copy()

    feature_columns = [
        column for column in encoded_features.columns
        if column != "client_id"
    ]

    scaler = StandardScaler()

    encoded_features[feature_columns] = scaler.fit_transform(
        encoded_features[feature_columns]
    )

    return encoded_features, scaler


# Step 5: Run Feature Engineering
def run_feature_engineering(clients):

    features = select_features(clients)

    features = handle_missing_values(features)

    encoded_features = encode_features(features)

    scaled_features, scaler = scale_features(encoded_features)

    print("FEATURE ENGINEERING COMPLETED!")

    print("Encoded Shape:", encoded_features.shape)

    print("Scaled Shape:", scaled_features.shape)

    return scaled_features, scaler


if __name__ == "__main__":

    from data_cleaning import load_datasets, clean_clients

    clients, properties = load_datasets()

    clients = clean_clients(clients)

    scaled_features, scaler = run_feature_engineering(clients)
