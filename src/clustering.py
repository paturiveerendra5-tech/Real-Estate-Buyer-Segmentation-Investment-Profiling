# CLUSTERING MODULE

import pandas as pd
from pathlib import Path

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score


# Step 1: Project Paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"


# Step 2: Load Scaled Features

def load_scaled_features():

    features = pd.read_csv(
        PROCESSED_PATH / "scaled_features.csv"
    )

    client_ids = pd.read_csv(
        PROCESSED_PATH / "client_ids.csv"
    )

    return features, client_ids


# Step 3: K-Means Clustering

def run_kmeans(features, clusters=4):

    model = KMeans(
        n_clusters=clusters,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(features)

    score = silhouette_score(features, labels)

    print("K-MEANS CLUSTERING")
    print("------------------")
    print("Number of Clusters:", clusters)
    print("Silhouette Score:", round(score, 4))

    return model, labels, score


# Step 4: Hierarchical Clustering

def run_hierarchical(features, clusters=2):

    model = AgglomerativeClustering(
        n_clusters=clusters,
        linkage="ward"
    )

    labels = model.fit_predict(features)

    score = silhouette_score(features, labels)

    print("\nHIERARCHICAL CLUSTERING")
    print("-----------------------")
    print("Number of Clusters:", clusters)
    print("Silhouette Score:", round(score, 4))

    return model, labels, score


# Step 5: Save Clustering Results

def save_results(client_ids, kmeans_labels, hierarchical_labels):

    results = pd.DataFrame({
        "client_id": client_ids.iloc[:, 0],
        "kmeans_cluster": kmeans_labels,
        "hierarchical_cluster": hierarchical_labels
    })

    results.to_csv(
        PROCESSED_PATH / "clustering_results.csv",
        index=False
    )

    print("\nClustering results saved successfully!")

    return results


# Step 6: Main Function

def main():

    features, client_ids = load_scaled_features()

    kmeans_model, kmeans_labels, kmeans_score = run_kmeans(
        features,
        clusters=4
    )

    hierarchical_model, hierarchical_labels, hierarchical_score = (
        run_hierarchical(features, clusters=2)
    )

    results = save_results(
        client_ids,
        kmeans_labels,
        hierarchical_labels
    )

    print("\nFinal Results Shape:", results.shape)


if __name__ == "__main__":

    main()
