import numpy as np


def choose_instances_from_clusters(clusters: np.ndarray) -> list[int]:
    """Randomly choose one instance from each cluster"""
    num_clusters = np.unique(clusters)
    indexes = [
        np.random.choice(np.where(clusters == cluster_ind)[0])
        for cluster_ind in num_clusters
    ]
    return indexes
