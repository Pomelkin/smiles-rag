from baseline.knowledge_base import QdrantKnowledgeBase
import numpy as np


class Drafter:
    def __init__(self):
        self._knowledge_base = QdrantKnowledgeBase()

    def retrieve_estimated_knowledge(self, query: str):
        retrieved_points = self._knowledge_base.get_similar_points(query)
        retrieved_vectors = np.array([point.vector for point in retrieved_points])
