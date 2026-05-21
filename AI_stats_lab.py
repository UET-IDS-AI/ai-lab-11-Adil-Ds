"""
AI_stats_lab.py

Lab: Unsupervised Learning and K-Means Clustering
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans


# ============================================================
# Question 1: Unlabeled Data and K-Means Clustering
# ============================================================

def load_iris_unlabeled(feature_indices=(0, 1)):
    """
    Load the Iris dataset without labels.
    """

    iris = load_iris()

    X = iris.data[:, feature_indices]

    feature_names = [
        iris.feature_names[i]
        for i in feature_indices
    ]

    return {
        "X": X,
        "feature_names": feature_names
    }


def standardize_features(X):
    """
    Standardize features to zero mean and unit variance.
    """

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    # Avoid division by zero
    std = np.where(std == 0, 1.0, std)

    X_scaled = (X - mean) / std

    return {
        "X_scaled": X_scaled,
        "mean": mean,
        "std": std
    }


def fit_kmeans(X, K, random_state=0, n_init=10):
    """
    Fit K-Means clustering on data X.
    """

    model = KMeans(
        n_clusters=K,
        random_state=random_state,
        n_init=n_init
    )

    model.fit(X)

    return {
        "centroids": model.cluster_centers_,
        "labels": model.labels_,
        "objective": model.inertia_,
        "n_iter": model.n_iter_
    }


def compute_kmeans_objective(X, centroids, labels):
    """
    Compute the K-Means objective manually.
    """

    objective = 0.0

    for i in range(len(X)):
        centroid = centroids[labels[i]]
        distance_squared = np.sum((X[i] - centroid) ** 2)
        objective += distance_squared

    return objective


# ============================================================
# Question 2: Choosing K, Underfitting/Overfitting, and Outliers
# ============================================================

def evaluate_k_values(X, k_values, random_state=0, n_init=10):
    """
    Run K-Means for multiple values of K.
    """

    objectives = []
    relative_improvements = []

    previous_objective = None

    for K in k_values:

        result = fit_kmeans(
            X,
            K,
            random_state=random_state,
            n_init=n_init
        )

        objective = result["objective"]

        objectives.append(objective)

        if previous_objective is None:
            relative_improvements.append(0.0)
        else:
            improvement = (
                previous_objective - objective
            ) / previous_objective

            relative_improvements.append(improvement)

        previous_objective = objective

    return {
        "k_values": k_values,
        "objectives": objectives,
        "relative_improvements": relative_improvements
    }


def choose_elbow_k(k_values, objectives):
    """
    Choose K using maximum-distance-to-line heuristic.
    """

    if len(k_values) < 3:
        return k_values[0]

    x = np.array(k_values, dtype=float)
    y = np.array(objectives, dtype=float)

    # First and last points
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])

    line_vector = p2 - p1
    line_length = np.linalg.norm(line_vector)

    if line_length == 0:
        return k_values[0]

    distances = []

    for i in range(len(x)):

        point = np.array([x[i], y[i]])

        # Perpendicular distance from point to line
        numerator = np.abs(
            np.cross(line_vector, point - p1)
        )

        distance = numerator / line_length

        distances.append(distance)

    best_index = int(np.argmax(distances))

    return k_values[best_index]


def cluster_size_summary(labels, K):
    """
    Count number of points in each cluster.
    """

    summary = {}

    for cluster_index in range(K):
        summary[cluster_index] = int(np.sum(labels == cluster_index))

    return summary


def identify_outliers_by_distance(X, centroids, labels, top_n=5):
    """
    Identify farthest points from assigned centroid.
    """

    distances = []

    for i in range(len(X)):

        centroid = centroids[labels[i]]

        distance_squared = np.sum((X[i] - centroid) ** 2)

        distances.append(distance_squared)

    distances = np.array(distances)

    sorted_indices = np.argsort(distances)[::-1]

    top_indices = sorted_indices[:top_n]

    return {
        "indices": top_indices,
        "distances": distances[top_indices]
    }


def diagnose_clustering_fit(K, elbow_k):
    """
    Diagnose clustering fit quality.
    """

    if K < elbow_k:
        return "underfitting"

    if K == elbow_k:
        return "good_fit"

    return "overfitting"


# ============================================================
# Question 3: Visualization
# ============================================================

def plot_unlabeled_data(X, feature_names=None, title="Unlabeled Data"):
    """
    Visualize unlabeled 2D data.
    """

    fig, ax = plt.subplots()

    ax.scatter(X[:, 0], X[:, 1])

    ax.set_title(title)

    if feature_names is not None:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])

    return fig, ax


def plot_kmeans_clusters(
    X,
    labels,
    centroids,
    feature_names=None,
    title="K-Means Clusters"
):
    """
    Visualize K-Means clustering results.
    """

    fig, ax = plt.subplots()

    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=labels
    )

    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=200,
        marker='X'
    )

    ax.set_title(title)

    if feature_names is not None:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])

    return fig, ax


def plot_elbow_curve(k_values, objectives, title="Elbow Method"):
    """
    Plot K-Means objective values versus K.
    """

    fig, ax = plt.subplots()

    ax.plot(k_values, objectives, marker='o')

    ax.set_title(title)
    ax.set_xlabel("Number of clusters K")
    ax.set_ylabel("Objective value")

    return fig, ax


if __name__ == "__main__":
    print("Implement all required functions.")
