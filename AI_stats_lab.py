"""
AI_stats_lab.py

Lab: Unsupervised Learning and K-Means Clustering

Topics:
- Unsupervised learning with unlabeled data
- Iris dataset without labels
- Feature standardization
- K-Means clustering
- K-Means objective function
- Elbow method for choosing K
- Underfitting and overfitting in clustering
- Distance-based outlier detection
- Visualization of unlabeled data, clusters, centroids, and elbow curve

Instructions:
- Implement all functions.
- Do NOT change function names.
- Do NOT print inside functions.
- Return exactly the required formats.
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

    Parameters:
        feature_indices : tuple
            Indices of features to use.
            Default (0, 1) means:
                sepal length
                sepal width

    Returns:
        A dictionary:

        {
            "X": feature matrix with selected columns,
            "feature_names": list of selected feature names
        }

    Notes:
        - Do NOT return the class labels.
        - This is an unsupervised learning setup.
    """
    iris = load_iris()
    X = iris.data[:, list(feature_indices)]
    feature_names = [iris.feature_names[i] for i in feature_indices]
    return {
        "X": X,
        "feature_names": feature_names
    }


def standardize_features(X):
    """
    Standardize features to zero mean and unit variance.

    Parameters:
        X : NumPy array of shape (n_samples, n_features)

    Returns:
        A dictionary:

        {
            "X_scaled": standardized feature matrix,
            "mean": feature-wise mean,
            "std": feature-wise standard deviation
        }

    Formula:
        X_scaled = (X - mean) / std

    Notes:
        - If any std value is 0, replace it with 1 before division.
        - This avoids division by zero.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    # Replace zero std with 1 to avoid division by zero
    std_safe = np.where(std == 0, 1, std)
    X_scaled = (X - mean) / std_safe
    return {
        "X_scaled": X_scaled,
        "mean": mean,
        "std": std
    }


def fit_kmeans(X, K, random_state=0, n_init=10):
    """
    Fit K-Means clustering on data X.

    Parameters:
        X            : feature matrix
        K            : number of clusters
        random_state : random seed
        n_init       : number of centroid initializations

    Returns:
        A dictionary:

        {
            "centroids": learned centroids,
            "labels": cluster assignment for each point,
            "objective": K-Means objective value,
            "n_iter": number of iterations used
        }

    Notes:
        - Use sklearn.cluster.KMeans.
        - The K-Means objective is the sum of squared distances
          from each point to its assigned centroid.
        - In sklearn, this value is stored in model.inertia_.
    """
    model = KMeans(n_clusters=K, random_state=random_state, n_init=n_init)
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

    Parameters:
        X         : feature matrix
        centroids : centroid matrix of shape (K, n_features)
        labels    : assigned cluster index for each point

    Returns:
        objective : sum of squared distances from each point
                    to its assigned centroid

    Formula:
        objective = sum_i || x_i - c_{label_i} ||^2
    """
    # For each point, get its assigned centroid and compute squared distance
    assigned_centroids = centroids[labels]          # shape (n_samples, n_features)
    diff = X - assigned_centroids                   # shape (n_samples, n_features)
    objective = np.sum(diff ** 2)
    return objective


# ============================================================
# Question 2: Choosing K, Underfitting/Overfitting, and Outliers
# ============================================================

def evaluate_k_values(X, k_values, random_state=0, n_init=10):
    """
    Run K-Means for multiple values of K.

    Parameters:
        X            : feature matrix
        k_values     : list of K values
        random_state : random seed
        n_init       : number of centroid initializations

    Returns:
        A dictionary:

        {
            "k_values": list of K values,
            "objectives": list of objective values,
            "relative_improvements": list of relative improvements
        }

    Relative improvement:
        For the first K, improvement is 0.0.
        For later K values:

        improvement = (previous_objective - current_objective) / previous_objective

    Notes:
        - Objective should usually decrease as K increases.
        - Very large K can overfit by creating too many small clusters.
    """
    objectives = []
    for K in k_values:
        result = fit_kmeans(X, K, random_state=random_state, n_init=n_init)
        objectives.append(result["objective"])

    relative_improvements = [0.0]
    for i in range(1, len(objectives)):
        prev = objectives[i - 1]
        curr = objectives[i]
        improvement = (prev - curr) / prev if prev != 0 else 0.0
        relative_improvements.append(improvement)

    return {
        "k_values": list(k_values),
        "objectives": objectives,
        "relative_improvements": relative_improvements
    }


def choose_elbow_k(k_values, objectives):
    """
    Choose K using a simple elbow heuristic.

    Parameters:
        k_values   : list of K values
        objectives : list of K-Means objective values

    Returns:
        best_k : K value at the elbow point

    Method:
        Use the maximum-distance-to-line heuristic.

        1. Treat the first and last points of the objective curve
           as endpoints of a straight line.
        2. Compute the perpendicular distance of each intermediate point
           from this line.
        3. Return the K value with the largest distance.

    Notes:
        - If fewer than 3 K values are given, return the first K.
        - This is a heuristic, not a perfect rule.
    """
    if len(k_values) < 3:
        return k_values[0]

    k_arr = np.array(k_values, dtype=float)
    obj_arr = np.array(objectives, dtype=float)

    # Normalize both axes to [0, 1] so that K scale and objective scale are comparable
    k_norm = (k_arr - k_arr[0]) / (k_arr[-1] - k_arr[0])
    obj_norm = (obj_arr - obj_arr[-1]) / (obj_arr[0] - obj_arr[-1]) if (obj_arr[0] - obj_arr[-1]) != 0 else obj_arr

    # Direction vector of the line from first to last point
    p1 = np.array([k_norm[0], obj_norm[0]])
    p2 = np.array([k_norm[-1], obj_norm[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    distances = []
    for i in range(len(k_values)):
        point = np.array([k_norm[i], obj_norm[i]])
        # Perpendicular distance from point to line p1-p2 (2D scalar cross product)
        v = p1 - point
        cross = np.abs(line_vec[0] * v[1] - line_vec[1] * v[0])
        dist = cross / line_len if line_len != 0 else 0.0
        distances.append(dist)

    best_idx = int(np.argmax(distances))
    return k_values[best_idx]


def cluster_size_summary(labels, K):
    """
    Count how many data points belong to each cluster.

    Parameters:
        labels : cluster assignment for each point
        K      : number of clusters

    Returns:
        A dictionary:

        {
            cluster_index: number_of_points_in_that_cluster
        }

    Example:
        labels = [0, 0, 1, 1, 1]
        K = 2

        output:
        {
            0: 2,
            1: 3
        }
    """
    summary = {}
    for k in range(K):
        summary[k] = int(np.sum(np.array(labels) == k))
    return summary


def identify_outliers_by_distance(X, centroids, labels, top_n=5):
    """
    Identify possible outliers based on distance from assigned centroid.

    Parameters:
        X         : feature matrix
        centroids : centroid matrix
        labels    : assigned cluster label for each point
        top_n     : number of farthest points to return

    Returns:
        A dictionary:

        {
            "indices": indices of top_n farthest points,
            "distances": squared distances of those points
        }

    Notes:
        - A point far from its assigned centroid may be unusual.
        - Sort points by distance in descending order.
        - Return the top_n farthest points.
    """
    assigned_centroids = centroids[labels]          # (n_samples, n_features)
    diff = X - assigned_centroids
    sq_distances = np.sum(diff ** 2, axis=1)        # (n_samples,)

    # Sort in descending order
    sorted_indices = np.argsort(sq_distances)[::-1]
    top_indices = sorted_indices[:top_n]
    top_distances = sq_distances[top_indices]

    return {
        "indices": top_indices,
        "distances": top_distances
    }


def diagnose_clustering_fit(K, elbow_k):
    """
    Diagnose whether the chosen K is likely underfitting, good fit, or overfitting.

    Parameters:
        K        : chosen number of clusters
        elbow_k  : elbow-method recommended K

    Returns:
        diagnosis string:

        if K < elbow_k:
            "underfitting"

        if K == elbow_k:
            "good_fit"

        if K > elbow_k:
            "overfitting"

    Notes:
        - In clustering, very small K may merge true groups.
        - Very large K may split meaningful groups into tiny clusters.
    """
    if K < elbow_k:
        return "underfitting"
    elif K == elbow_k:
        return "good_fit"
    else:
        return "overfitting"


# ============================================================
# Question 3: Visualization
# ============================================================

def plot_unlabeled_data(X, feature_names=None, title="Unlabeled Data"):
    """
    Visualize unlabeled 2D data.

    Parameters:
        X             : feature matrix of shape (n_samples, 2)
        feature_names : optional list of two feature names
        title         : plot title

    Returns:
        fig, ax

    Notes:
        - This function should create a scatter plot.
        - Do not use labels/classes because this is unsupervised learning.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(X[:, 0], X[:, 1], color="steelblue", alpha=0.6, edgecolors="white", linewidths=0.4, s=50)
    ax.set_title(title)
    if feature_names is not None and len(feature_names) >= 2:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
    else:
        ax.set_xlabel("Feature 0")
        ax.set_ylabel("Feature 1")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig, ax


def plot_kmeans_clusters(X, labels, centroids, feature_names=None, title="K-Means Clusters"):
    """
    Visualize K-Means clustering results.

    Parameters:
        X             : feature matrix of shape (n_samples, 2)
        labels        : cluster assignment for each point
        centroids     : learned cluster centroids
        feature_names : optional list of two feature names
        title         : plot title

    Returns:
        fig, ax

    Notes:
        - Plot data points colored by cluster label.
        - Plot centroids using a large marker.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(unique_labels)))

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            X[mask, 0], X[mask, 1],
            color=colors[idx],
            alpha=0.65,
            edgecolors="white",
            linewidths=0.4,
            s=55,
            label=f"Cluster {label}"
        )

    # Plot centroids
    ax.scatter(
        centroids[:, 0], centroids[:, 1],
        color="black",
        marker="X",
        s=200,
        zorder=5,
        label="Centroids",
        edgecolors="white",
        linewidths=0.8
    )

    ax.set_title(title)
    if feature_names is not None and len(feature_names) >= 2:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
    else:
        ax.set_xlabel("Feature 0")
        ax.set_ylabel("Feature 1")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig, ax


def plot_elbow_curve(k_values, objectives, title="Elbow Method"):
    """
    Plot K-Means objective values versus K.

    Parameters:
        k_values   : list of K values
        objectives : list of objective values
        title      : plot title

    Returns:
        fig, ax

    Notes:
        - X-axis should show K.
        - Y-axis should show objective value.
        - This plot helps identify the elbow point.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(k_values, objectives, marker="o", color="steelblue", linewidth=2, markersize=7, markerfacecolor="white", markeredgewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("K-Means Objective (Inertia)")
    ax.set_xticks(k_values)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    print("Implement all required functions.")
