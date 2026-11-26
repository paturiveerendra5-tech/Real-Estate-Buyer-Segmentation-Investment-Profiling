# ============================================================
# CLUSTER ANALYSIS MODULE
# Real Estate Buyer Segmentation & Investment Profiling
# ============================================================

import pandas as pd
from pathlib import Path


# Step 1: Project Paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FINAL_PATH = PROJECT_ROOT / "data" / "final"
REPORT_PATH = PROJECT_ROOT / "reports"

REPORT_PATH.mkdir(parents=True, exist_ok=True)


# Step 2: Load Final Buyer Dataset

def load_final_data():

    file_path = FINAL_PATH / "final_buyer_segmentation.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    buyers = pd.read_csv(file_path)

    if buyers.empty:
        raise ValueError("Final buyer dataset is empty!")

    print("\nFinal Buyer Dataset Loaded Successfully!")
    print("Dataset Shape:", buyers.shape)

    return buyers


# Step 3: Validate Required Columns

def validate_columns(buyers):

    required_columns = [
        "client_id",
        "country",
        "region",
        "satisfaction_score",
        "client_type",
        "acquisition_purpose",
        "loan_applied"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in buyers.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


# Step 4: Identify Cluster Column

def get_cluster_column(buyers):

    if "cluster" in buyers.columns:
        return "cluster"

    elif "kmeans_cluster" in buyers.columns:
        return "kmeans_cluster"

    else:
        raise ValueError(
            "Cluster column not found in dataset!"
        )


# Step 5: Cluster Summary

def generate_cluster_summary(buyers):

    summary = buyers.groupby("cluster").agg(

        total_buyers=("client_id", "count"),

        countries=("country", "nunique"),

        regions=("region", "nunique"),

        average_satisfaction=(
            "satisfaction_score",
            "mean"
        )

    ).reset_index()

    summary["average_satisfaction"] = (
        summary["average_satisfaction"].round(2)
    )

    summary["buyer_percentage"] = (
        summary["total_buyers"]
        / summary["total_buyers"].sum()
        * 100
    ).round(2)

    return summary


# Step 6: Buyer Type Analysis

def analyze_buyer_types(buyers):

    analysis = pd.crosstab(
        buyers["cluster"],
        buyers["client_type"],
        margins=True
    )

    return analysis


# Step 7: Acquisition Purpose Analysis

def analyze_acquisition_purpose(buyers):

    analysis = pd.crosstab(
        buyers["cluster"],
        buyers["acquisition_purpose"],
        margins=True
    )

    return analysis


# Step 8: Loan Application Analysis

def analyze_loan_behavior(buyers):

    analysis = pd.crosstab(
        buyers["cluster"],
        buyers["loan_applied"],
        margins=True
    )

    return analysis


# Step 9: Geographic Analysis

def analyze_geography(buyers):

    country_analysis = pd.crosstab(
        buyers["cluster"],
        buyers["country"],
        margins=True
    )

    region_analysis = pd.crosstab(
        buyers["cluster"],
        buyers["region"],
        margins=True
    )

    return country_analysis, region_analysis


# Step 10: Save Analysis Results

def save_analysis_results(
    summary,
    buyer_types,
    acquisition,
    loan,
    countries,
    regions
):

    results = {

        "cluster_analysis_summary.csv": summary,

        "buyer_type_analysis.csv": buyer_types,

        "acquisition_purpose_analysis.csv": acquisition,

        "loan_behavior_analysis.csv": loan,

        "country_analysis.csv": countries,

        "region_analysis.csv": regions

    }

    for filename, dataframe in results.items():

        output_file = REPORT_PATH / filename

        dataframe.to_csv(output_file)

        print(f"Saved Successfully: {filename}")


# Step 11: Main Function

def main():

    print("\n================================")
    print(" REAL ESTATE CLUSTER ANALYSIS")
    print("================================")

    # Load Dataset

    buyers = load_final_data()

    # Validate Dataset

    validate_columns(buyers)

    # Identify Cluster Column

    cluster_column = get_cluster_column(buyers)

    if cluster_column != "cluster":

        buyers = buyers.rename(
            columns={cluster_column: "cluster"}
        )

    # Handle Missing Values in Cluster Column

    buyers = buyers.dropna(subset=["cluster"]).copy()

    if buyers.empty:
        raise ValueError(
            "No valid cluster records found!"
        )

    # Convert Satisfaction Score to Numeric

    buyers["satisfaction_score"] = pd.to_numeric(
        buyers["satisfaction_score"],
        errors="coerce"
    )

    # Generate Cluster Summary

    summary = generate_cluster_summary(buyers)

    # Buyer Analysis

    buyer_types = analyze_buyer_types(buyers)

    acquisition = analyze_acquisition_purpose(buyers)

    loan = analyze_loan_behavior(buyers)

    # Geographic Analysis

    countries, regions = analyze_geography(buyers)

    # Display Results

    print("\n1. CLUSTER SUMMARY")
    print(summary.to_string(index=False))

    print("\n2. BUYER TYPE ANALYSIS")
    print(buyer_types)

    print("\n3. ACQUISITION PURPOSE ANALYSIS")
    print(acquisition)

    print("\n4. LOAN APPLICATION ANALYSIS")
    print(loan)

    print("\n5. COUNTRY ANALYSIS")
    print(countries)

    print("\n6. REGION ANALYSIS")
    print(regions)

    # Save All Results

    save_analysis_results(
        summary,
        buyer_types,
        acquisition,
        loan,
        countries,
        regions
    )

    print("\n================================")
    print(" CLUSTER ANALYSIS COMPLETED!")
    print("================================")


# Step 12: Execute Main Function

if __name__ == "__main__":

    main()