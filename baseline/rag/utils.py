import numpy as np
from sklearn.metrics import euclidean_distances
from scipy.special import softmax
from dataclasses import dataclass
from qdrant_client import models


def choose_instances_from_clusters(clusters: np.ndarray) -> list[int]:
    """Select instances from clusters based on cosine similarity and randomness.
    The function works as follows:
    1. Select the top-1 vector from 9 clusters with the closest cosine distance to the input question.
    2. From each of the remaining 2 clusters, select a random vector."""

    top_1_cluster = clusters[0]
    num_clusters = np.unique(clusters[clusters != top_1_cluster])

    # select random vectors from each cluster
    indexes = [
        np.random.choice(np.where(clusters == cluster_ind)[0])
        for cluster_ind in num_clusters
    ]

    indexes.insert(0, 0)  # insert top-1 vector (0)
    return indexes


def calculate_centroids(clusters: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Calculate the centroids of each cluster."""
    centroids = np.zeros((len(np.unique(clusters)), vectors.shape[1]))
    for cluster_ind in np.unique(clusters):
        centroids[cluster_ind] = vectors[cluster_ind == clusters].mean(axis=0)
    return centroids


def calculate_centroid_distance(
    vectors: np.ndarray, centroids: np.ndarray, use_softmax: bool = False
) -> np.ndarray:
    """Calculate the distance between each vector and its centroid."""
    distances = np.zeros(vectors.shape[0])
    for i, (vector, centroid) in enumerate(zip(vectors, centroids)):
        distances[i] = euclidean_distances(vector[None, :], centroid[None, :])[0]

    if use_softmax:
        distances = softmax(distances, axis=0)
    return distances


@dataclass
class EstimatedPoint:
    point: models.ScoredPoint
    distance: float
