from baseline.knowledge_base import QdrantKnowledgeBase
import numpy as np


class Drafter:
    def __init__(self):
        self._knowledge_base = QdrantKnowledgeBase()

    def __call__(self, query: str):
        """Get similar points with an estimation based on Lowe's score for the top 1 and the distance to the centroid of each cluster"""
        retrieved_points, embedding = self._knowledge_base.get_similar_points(query)
        retrieved_vectors = np.array([point.vector for point in retrieved_points])
        return retrieved_points, embedding
