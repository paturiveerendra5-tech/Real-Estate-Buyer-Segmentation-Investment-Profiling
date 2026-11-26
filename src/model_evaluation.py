# ============================================================
# MODEL EVALUATION MODULE
# Real Estate Buyer Segmentation & Investment Profiling
# ============================================================

import pandas as pd

from pathlib import Path

from sklearn.cluster import KMeans, AgglomerativeClustering

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


# Step 1: Project Paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = PROJECT_ROOT / "reports"

PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
REPORT_PATH.mkdir(parents=True, exist_ok=True)


# Step 2: Load Scaled Features

def load_scaled_features():

    file_path = PROCESSED_PATH / "scaled_features.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Scaled features not found: {file_path}"
        )

    features = pd.read_csv(file_path)

    if features.empty:
        raise ValueError("Scaled features dataset is empty!")

    if features.isnull().any().any():
        raise ValueError("Missing values found in features!")

    if not all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in features.dtypes
    ):
        raise ValueError("All features must be numeric!")

    if not features.map(
        lambda x: pd.notna(x) and abs(x) != float("inf")
    ).all().all():
        raise ValueError("Infinite values found in features!")

    if len(features) < 3:
        raise ValueError(
            "At least 3 samples are required for evaluation!"
        )

    print("\nScaled Features Loaded Successfully!")
    print("Features Shape:", features.shape)

    return features


# Step 3: Calculate Evaluation Metrics

def calculate_metrics(features, labels):

    unique_clusters = len(set(labels))

    if unique_clusters < 2 or unique_clusters >= len(features):
        raise ValueError(
            "Invalid number of clusters for evaluation!"
        )

    silhouette = silhouette_score(features, labels)

    davies_bouldin = davies_bouldin_score(
        features,
        labels
    )

    calinski_harabasz = calinski_harabasz_score(
        features,
        labels
    )

    return {
        "Silhouette Score": round(silhouette, 4),

        "Davies-Bouldin Score": round(davies_bouldin, 4),

        "Calinski-Harabasz Score": round(
            calinski_harabasz, 4
        )
    }


# Step 4: Evaluate K-Means

def evaluate_kmeans(features, clusters=4):

    if clusters < 2 or clusters >= len(features):
        raise ValueError("Invalid K-Means cluster count!")

    model = KMeans(
        n_clusters=clusters,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(features)

    metrics = calculate_metrics(
        features,
        labels
    )

    return metrics


# Step 5: Evaluate Hierarchical Clustering

def evaluate_hierarchical(features, clusters=2):

    if clusters < 2 or clusters >= len(features):
        raise ValueError("Invalid Hierarchical cluster count!")

    model = AgglomerativeClustering(
        n_clusters=clusters,
        linkage="ward"
    )

    labels = model.fit_predict(features)

    metrics = calculate_metrics(
        features,
        labels
    )

    return metrics


# Step 6: Find Best K-Means Cluster Count

def evaluate_multiple_clusters(features):

    results = []

    max_clusters = min(8, len(features) - 1)

    for clusters in range(2, max_clusters + 1):

        metrics = evaluate_kmeans(
            features,
            clusters
        )

        results.append({

            "Model": "K-Means",

            "Number of Clusters": clusters,

            **metrics

        })

    comparison = pd.DataFrame(results)

    return comparison


# Step 7: Compare Models

def compare_models(features):

    kmeans_metrics = evaluate_kmeans(
        features,
        clusters=4
    )

    hierarchical_metrics = evaluate_hierarchical(
        features,
        clusters=2
    )

    comparison = pd.DataFrame([

        {
            "Model": "K-Means",
            "Number of Clusters": 4,
            **kmeans_metrics
        },

        {
            "Model": "Hierarchical",
            "Number of Clusters": 2,
            **hierarchical_metrics
        }

    ])

    return comparison


# Step 8: Save Evaluation Results

def save_evaluation(
    comparison,
    cluster_results
):

    comparison.to_csv(
        REPORT_PATH / "model_evaluation_comparison.csv",
        index=False
    )

    cluster_results.to_csv(
        REPORT_PATH / "kmeans_cluster_optimization.csv",
        index=False
    )

    print("\nEvaluation Reports Saved Successfully!")


# Step 9: Main Function

def main():

    print("\n====================================")
    print(" REAL ESTATE MODEL EVALUATION")
    print("====================================")

    # Load Features

    features = load_scaled_features()

    # Compare Models

    comparison = compare_models(features)

    # Test Multiple Cluster Counts

    cluster_results = evaluate_multiple_clusters(
        features
    )

    # Display Model Comparison

    print("\nMODEL COMPARISON")
    print("----------------")

    print(comparison.to_string(index=False))

    # Display Cluster Optimization

    print("\nK-MEANS CLUSTER OPTIMIZATION")
    print("----------------------------")

    print(cluster_results.to_string(index=False))

    # Save Reports

    save_evaluation(
        comparison,
        cluster_results
    )

    print("\n====================================")
    print(" MODEL EVALUATION COMPLETED!")
    print("====================================")


# Step 10: Execute Main Function

if __name__ == "__main__":

    main()