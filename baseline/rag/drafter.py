from sklearn.cluster import KMeans
from baseline.knowledge_base import QdrantKnowledgeBase
import numpy as np
from .utils import choose_instances_from_clusters, calculate_centroids
from qdrant_client import models


class Drafter:
    def __init__(self):
        self._knowledge_base = QdrantKnowledgeBase()

    def __call__(self, query: str):
        """Get similar points with an estimation based on Lowe's score for the top 1 and the distance to the centroid of each cluster"""
        retrieved_points, embedding = self._knowledge_base.get_similar_points(query)
        retrieved_vectors = np.array([point.vector for point in retrieved_points])
        return retrieved_points, embedding

    @staticmethod
    def estimate_points(points: list[models.ScoredPoint]):
        vectors = np.array([point.vector for point in points])
        clusters = KMeans(n_clusters=3).fit_predict(vectors)
        ind = choose_instances_from_clusters(clusters)
        centroids = calculate_centroids(clusters, vectors)
