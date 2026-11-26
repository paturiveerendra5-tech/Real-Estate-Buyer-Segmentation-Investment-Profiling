# ============================================================
# DATA CLEANING MODULE
# Real Estate Buyer Segmentation & Investment Profiling
# ============================================================

import pandas as pd
from pathlib import Path


# Step 1: Project Paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data"

PROCESSED_PATH = DATA_PATH / "processed"

PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

CLIENTS_FILE = DATA_PATH / "clients.csv"
PROPERTIES_FILE = DATA_PATH / "properties.csv"


# Step 2: Load Datasets

def load_datasets():

    if not CLIENTS_FILE.exists():
        raise FileNotFoundError(
            f"Clients dataset not found: {CLIENTS_FILE}"
        )

    if not PROPERTIES_FILE.exists():
        raise FileNotFoundError(
            f"Properties dataset not found: {PROPERTIES_FILE}"
        )

    clients = pd.read_csv(CLIENTS_FILE)

    properties = pd.read_csv(PROPERTIES_FILE)

    if clients.empty or properties.empty:
        raise ValueError("Input datasets cannot be empty!")

    print("\nDatasets Loaded Successfully!")

    print("Clients Shape:", clients.shape)

    print("Properties Shape:", properties.shape)

    return clients, properties


# Step 3: Validate Required Columns

def validate_columns(clients, properties):

    required_client_columns = [
        "client_id",
        "date_of_birth",
        "gender",
        "client_type"
    ]

    required_property_columns = [
        "sale_price",
        "transaction_date"
    ]

    missing_clients = [
        col for col in required_client_columns
        if col not in clients.columns
    ]

    missing_properties = [
        col for col in required_property_columns
        if col not in properties.columns
    ]

    if missing_clients:
        raise ValueError(
            f"Missing client columns: {missing_clients}"
        )

    if missing_properties:
        raise ValueError(
            f"Missing property columns: {missing_properties}"
        )


# Step 4: Clean Client Data

def clean_clients(clients):

    clients = clients.copy()

    # Convert Date of Birth

    clients["date_of_birth"] = pd.to_datetime(
        clients["date_of_birth"],
        errors="coerce",
        dayfirst=True
    )

    # Clean Text Columns

    for column in ["gender", "client_type"]:

        clients[column] = (
            clients[column]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
        )

    # Convert Categorical Columns

    clients["gender"] = clients["gender"].astype("category")

    clients["client_type"] = clients["client_type"].astype("category")

    # Remove Duplicate Client Records

    clients = clients.drop_duplicates(
        subset=["client_id"]
    )

    return clients


# Step 5: Clean Property Data

def clean_properties(properties):

    properties = properties.copy()

    # Convert Sale Price to Numeric

    properties["sale_price"] = pd.to_numeric(

        properties["sale_price"]
        .astype(str)
        .str.replace(r"[$,]", "", regex=True)
        .str.strip(),

        errors="coerce"

    )

    # Convert Transaction Date

    properties["transaction_date"] = pd.to_datetime(
        properties["transaction_date"],
        errors="coerce",
        dayfirst=True
    )

    # Remove Negative Sale Prices

    properties.loc[
        properties["sale_price"] < 0,
        "sale_price"
    ] = pd.NA

    # Remove Duplicate Property Records

    properties = properties.drop_duplicates()

    return properties


# Step 6: Handle Missing Values Report

def generate_missing_values_report(clients, properties):

    clients_missing = clients.isnull().sum()

    properties_missing = properties.isnull().sum()

    missing_report = pd.concat(

        [
            clients_missing.rename("Clients Missing Values"),

            properties_missing.rename("Properties Missing Values")

        ],

        axis=1

    ).fillna(0).astype(int)

    output_file = PROCESSED_PATH / "missing_values_report.csv"

    missing_report.to_csv(output_file)

    print("\nMissing Values Report Saved!")

    return missing_report


# Step 7: Save Cleaned Datasets

def save_cleaned_data(clients, properties):

    clients.to_csv(
        PROCESSED_PATH / "cleaned_clients.csv",
        index=False
    )

    properties.to_csv(
        PROCESSED_PATH / "cleaned_properties.csv",
        index=False
    )

    print("\nCleaned Datasets Saved Successfully!")


# Step 8: Main Cleaning Function

def run_data_cleaning():

    print("\n====================================")
    print(" REAL ESTATE DATA CLEANING")
    print("====================================")

    # Load Data

    clients, properties = load_datasets()

    # Validate Columns

    validate_columns(clients, properties)

    # Clean Client Data

    clients = clean_clients(clients)

    # Clean Property Data

    properties = clean_properties(properties)

    # Generate Missing Values Report

    missing_report = generate_missing_values_report(
        clients,
        properties
    )

    # Save Cleaned Data

    save_cleaned_data(
        clients,
        properties
    )

    # Display Final Information

    print("\nDATA CLEANING SUMMARY")
    print("---------------------")

    print("Cleaned Clients Shape:", clients.shape)

    print("Cleaned Properties Shape:", properties.shape)

    print("\nMissing Values Summary:")
    print(missing_report)

    print("\n====================================")
    print(" DATA CLEANING COMPLETED!")
    print("====================================")

    return clients, properties


# Step 9: Execute Main Function

if __name__ == "__main__":

    run_data_cleaning()