import pandas as pd
from pathlib import Path

print("=" * 60)
print(" REAL ESTATE BUYER SEGMENTATION PROJECT")
print("=" * 60)

# Dataset paths
clients_path = Path("data/clients.csv")
properties_path = Path("data/properties.csv")


# Check dataset files
if not clients_path.exists():
    raise FileNotFoundError("clients.csv not found in data folder!")

if not properties_path.exists():
    raise FileNotFoundError("properties.csv not found in data folder!")


# Load Clients Dataset
clients = pd.read_csv(clients_path)

# Load Properties Dataset
properties = pd.read_csv(properties_path)


# Check whether files contain actual data
for name, df in [
    ("Clients", clients),
    ("Properties", properties)
]:
    if df.empty or df.shape[1] == 1 and str(df.columns[0]).startswith("C:\\"):
        raise ValueError(
            f"{name} dataset does not contain valid CSV data. "
            "Please check the original downloaded file."
        )


# Clients Dataset Information
print("\n1. CLIENTS DATASET")
print("-" * 40)

print("Rows:", clients.shape[0])
print("Columns:", clients.shape[1])
print("Shape:", clients.shape)

print("\nColumn Names:")
print(clients.columns.tolist())

print("\nFirst 5 Records:")
print(clients.head())

print("\nMissing Values:")
print(clients.isnull().sum())


# Properties Dataset Information
print("\n2. PROPERTIES DATASET")
print("-" * 40)

print("Rows:", properties.shape[0])
print("Columns:", properties.shape[1])
print("Shape:", properties.shape)

print("\nColumn Names:")
print(properties.columns.tolist())

print("\nFirst 5 Records:")
print(properties.head())

print("\nMissing Values:")
print(properties.isnull().sum())


print("\n" + "=" * 60)
print(" DATA LOADING PROCESS COMPLETED")
print("=" * 60)

# STEP 2: RUN CLUSTER ANALYSIS

from src.cluster_analysis import main as run_cluster_analysis
from src.model_evaluation import main as run_model_evaluation

print("\n3. CLUSTER ANALYSIS")
print("-" * 40)

run_cluster_analysis()


# STEP 3: MODEL EVALUATION

print("\n4. MODEL EVALUATION")
print("-" * 40)

run_model_evaluation()


# FINAL MESSAGE

print("\n" + "=" * 60)
print(" COMPLETE PROJECT EXECUTION FINISHED!")
print("=" * 60)