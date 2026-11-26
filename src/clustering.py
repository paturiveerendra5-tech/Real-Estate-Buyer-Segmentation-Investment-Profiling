# ============================================================
# REAL ESTATE BUYER SEGMENTATION
# CLUSTERING MODULE
# ============================================================

import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score


# ------------------------------------------------------------
# STEP 1: PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"

FEATURES_FILE = PROCESSED_PATH / "scaled_features.csv"
CLIENT_IDS_FILE = PROCESSED_PATH / "client_ids.csv"

RESULTS_FILE = PROCESSED_PATH / "clustering_results.csv"


# ------------------------------------------------------------
# STEP 2: LOAD DATA
# ------------------------------------------------------------

def load_scaled_features():

    if not FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"Scaled features file not found: {FEATURES_FILE}"
        )

    if not CLIENT_IDS_FILE.exists():
        raise FileNotFoundError(
            f"Client IDs file not found: {CLIENT_IDS_FILE}"
        )

    features = pd.read_csv(FEATURES_FILE)
    client_ids = pd.read_csv(CLIENT_IDS_FILE)

    if features.empty:
        raise ValueError("Scaled features dataset is empty.")

    if client_ids.empty:
        raise ValueError("Client IDs dataset is empty.")

    if len(features) != len(client_ids):
        raise ValueError(
            "Mismatch between features and client IDs."
        )

    # Keep only numeric feature columns
    features = features.select_dtypes(include=[np.number])

    if features.empty:
        raise ValueError("No numeric features available.")

    # Check missing and infinite values
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError(
            "Features contain missing or infinite values. "
            "Please clean the dataset first."
        )

    print("DATA LOADED SUCCESSFULLY")
    print("------------------------")
    print("Total Buyers:", len(features))
    print("Total Features:", features.shape[1])

    return features, client_ids


# ------------------------------------------------------------
# STEP 3: VALIDATE CLUSTER COUNT
# ------------------------------------------------------------

def validate_clusters(features, clusters):

    samples = len(features)

    if clusters < 2:
        raise ValueError("Number of clusters must be at least 2.")

    if clusters >= samples:
        raise ValueError(
            "Number of clusters must be smaller than number of buyers."
        )


# ------------------------------------------------------------
# STEP 4: K-MEANS CLUSTERING
# ------------------------------------------------------------

def run_kmeans(features, clusters=4):

    validate_clusters(features, clusters)

    model = KMeans(
        n_clusters=clusters,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(features)

    score = silhouette_score(features, labels)

    print("\nK-MEANS CLUSTERING")
    print("------------------")
    print("Number of Clusters:", clusters)
    print("Silhouette Score:", round(score, 4))

    return model, labels, score


# ------------------------------------------------------------
# STEP 5: HIERARCHICAL CLUSTERING
# ------------------------------------------------------------

def run_hierarchical(features, clusters=2):

    validate_clusters(features, clusters)

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


# ------------------------------------------------------------
# STEP 6: SAVE CLUSTERING RESULTS
# ------------------------------------------------------------

def save_results(client_ids, kmeans_labels, hierarchical_labels):

    results = pd.DataFrame({

        "client_id": client_ids.iloc[:, 0],

        "kmeans_cluster": kmeans_labels,

        "hierarchical_cluster": hierarchical_labels

    })

    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        RESULTS_FILE,
        index=False
    )

    print("\nCLUSTERING RESULTS")
    print("------------------")
    print("Total Buyers:", len(results))
    print("K-Means Groups:", results["kmeans_cluster"].nunique())
    print(
        "Hierarchical Groups:",
        results["hierarchical_cluster"].nunique()
    )

    print("\nResults saved successfully!")
    print("File:", RESULTS_FILE)

    return results


# ------------------------------------------------------------
# STEP 7: MAIN FUNCTION
# ------------------------------------------------------------

def main():

    print("=" * 55)
    print("REAL ESTATE BUYER SEGMENTATION")
    print("CLUSTERING MODULE")
    print("=" * 55)

    # Load data
    features, client_ids = load_scaled_features()

    # Run K-Means
    kmeans_model, kmeans_labels, kmeans_score = run_kmeans(
        features,
        clusters=4
    )

    # Run Hierarchical Clustering
    hierarchical_model, hierarchical_labels, hierarchical_score = (
        run_hierarchical(
            features,
            clusters=2
        )
    )

    # Save results
    results = save_results(
        client_ids,
        kmeans_labels,
        hierarchical_labels
    )

    # Final comparison
    print("\nMODEL COMPARISON")
    print("----------------")
    print("K-Means Silhouette Score:", round(kmeans_score, 4))
    print(
        "Hierarchical Silhouette Score:",
        round(hierarchical_score, 4)
    )

    if kmeans_score > hierarchical_score:
        print("\nBetter Silhouette Score: K-Means")
    elif hierarchical_score > kmeans_score:
        print("\nBetter Silhouette Score: Hierarchical")
    else:
        print("\nBoth models have equal Silhouette Scores.")

    print("\nFinal Results Shape:", results.shape)

    print("\nCLUSTERING COMPLETED SUCCESSFULLY!")


# ------------------------------------------------------------
# STEP 8: EXECUTION
# ------------------------------------------------------------

if __name__ == "__main__":

    main()

    