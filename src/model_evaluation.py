# MODEL EVALUATION MODULE

import pandas as pd
from pathlib import Path

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering


# Step 1: Project Paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"

FINAL_PATH = PROJECT_ROOT / "data" / "final"


# Step 2: Load Scaled Features

def load_scaled_features():

    features = pd.read_csv(
        PROCESSED_PATH / "scaled_features.csv"
    )

    print("Scaled Features Loaded Successfully!")

    print("Features Shape:", features.shape)

    return features


# Step 3: Evaluate K-Means

def evaluate_kmeans(features, clusters=4):

    model = KMeans(
        n_clusters=clusters,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(features)

    score = silhouette_score(features, labels)

    return score


# Step 4: Evaluate Hierarchical Clustering

def evaluate_hierarchical(features, clusters=2):

    model = AgglomerativeClustering(
        n_clusters=clusters,
        linkage="ward"
    )

    labels = model.fit_predict(features)

    score = silhouette_score(features, labels)

    return score


# Step 5: Model Comparison

def compare_models(features):

    kmeans_score = evaluate_kmeans(features, 4)

    hierarchical_score = evaluate_hierarchical(features, 2)

    comparison = pd.DataFrame({
        "Model": [
            "K-Means",
            "Hierarchical"
        ],
        "Number of Clusters": [
            4,
            2
        ],
        "Silhouette Score": [
            kmeans_score,
            hierarchical_score
        ]
    })

    return comparison


# Step 6: Save Evaluation Results

def save_evaluation(comparison):

    output_file = PROCESSED_PATH / "src_model_evaluation.csv"

    comparison.to_csv(
        output_file,
        index=False
    )

    print("\nEvaluation Results Saved Successfully!")


# Step 7: Main Function

def main():

    features = load_scaled_features()

    comparison = compare_models(features)

    print("\nMODEL EVALUATION RESULTS")
    print("------------------------")

    print(comparison.to_string(index=False))

    save_evaluation(comparison)

    print("\nMODEL EVALUATION COMPLETED!")


if __name__ == "__main__":

    main()