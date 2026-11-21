# DATA CLEANING MODULE

import pandas as pd
from pathlib import Path


# Project Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data"

CLIENTS_FILE = DATA_PATH / "clients.csv"
PROPERTIES_FILE = DATA_PATH / "properties.csv"


# Load Datasets
def load_datasets():

    clients = pd.read_csv(CLIENTS_FILE)
    properties = pd.read_csv(PROPERTIES_FILE)

    return clients, properties


# Clean Client Data
def clean_clients(clients):

    clients = clients.copy()

    clients["date_of_birth"] = pd.to_datetime(
        clients["date_of_birth"],
        errors="coerce",
        dayfirst=True
    )

    clients["gender"] = clients["gender"].astype("category")

    clients["client_type"] = clients["client_type"].astype("category")

    return clients


# Clean Property Data
def clean_properties(properties):

    properties = properties.copy()

    properties["sale_price"] = pd.to_numeric(
        properties["sale_price"]
        .astype(str)
        .str.replace(r"[$,]", "", regex=True),
        errors="coerce"
    )

    properties["transaction_date"] = pd.to_datetime(
        properties["transaction_date"],
        errors="coerce",
        dayfirst=True
    )

    return properties


# Main Cleaning Function
def run_data_cleaning():

    clients, properties = load_datasets()

    clients = clean_clients(clients)

    properties = clean_properties(properties)

    print("DATA CLEANING COMPLETED!")

    print("Clients Shape:", clients.shape)
    print("Properties Shape:", properties.shape)

    return clients, properties


if __name__ == "__main__":

    run_data_cleaning()