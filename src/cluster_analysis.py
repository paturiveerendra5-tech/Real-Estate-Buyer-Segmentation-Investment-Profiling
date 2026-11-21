# CLUSTER ANALYSIS MODULE

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

    buyers = pd.read_csv(file_path)

    print("Final Buyer Dataset Loaded Successfully!")

    print("Dataset Shape:", buyers.shape)

    return buyers


# Step 3: Identify Cluster Column

def get_cluster_column(buyers):

    if "cluster" in buyers.columns:
        return "cluster"

    elif "kmeans_cluster" in buyers.columns:
        return "kmeans_cluster"

    else:
        raise ValueError("Cluster column not found!")


# Step 4: Cluster Summary

def generate_cluster_summary(buyers):

    cluster_column = get_cluster_column(buyers)

    summary = buyers.groupby(cluster_column).agg(
        total_buyers=("client_id", "count"),
        countries=("country", "nunique"),
        regions=("region", "nunique"),
        average_satisfaction=("satisfaction_score", "mean")
    ).reset_index()

    summary = summary.rename(
        columns={cluster_column: "cluster"}
    )

    return summary


# Step 5: Buyer Type Analysis

def analyze_buyer_types(buyers):

    return pd.crosstab(
        buyers["cluster"],
        buyers["client_type"]
    )


# Step 6: Acquisition Purpose Analysis

def analyze_acquisition_purpose(buyers):

    return pd.crosstab(
        buyers["cluster"],
        buyers["acquisition_purpose"]
    )


# Step 7: Loan Application Analysis

def analyze_loan_behavior(buyers):

    return pd.crosstab(
        buyers["cluster"],
        buyers["loan_applied"]
    )


# Step 8: Geographic Analysis

def analyze_geography(buyers):

    country_analysis = pd.crosstab(
        buyers["cluster"],
        buyers["country"]
    )

    region_analysis = pd.crosstab(
        buyers["cluster"],
        buyers["region"]
    )

    return country_analysis, region_analysis


# Step 9: Save Analysis Results

def save_analysis_results(summary):

    output_file = REPORT_PATH / "cluster_analysis_summary.csv"

    summary.to_csv(output_file, index=False)

    print("Cluster Analysis Summary Saved Successfully!")


# Step 10: Main Function

def main():

    buyers = load_final_data()

    cluster_column = get_cluster_column(buyers)

    if cluster_column != "cluster":
        buyers = buyers.rename(
            columns={cluster_column: "cluster"}
        )

    summary = generate_cluster_summary(buyers)

    buyer_types = analyze_buyer_types(buyers)

    acquisition = analyze_acquisition_purpose(buyers)

    loan = analyze_loan_behavior(buyers)

    countries, regions = analyze_geography(buyers)

    print("\nCLUSTER SUMMARY")
    print(summary)

    print("\nBUYER TYPE ANALYSIS")
    print(buyer_types)

    print("\nACQUISITION PURPOSE ANALYSIS")
    print(acquisition)

    print("\nLOAN APPLICATION ANALYSIS")
    print(loan)

    print("\nCOUNTRY ANALYSIS")
    print(countries)

    print("\nREGION ANALYSIS")
    print(regions)

    save_analysis_results(summary)

    print("\nCLUSTER ANALYSIS COMPLETED!")


if __name__ == "__main__":

    main()