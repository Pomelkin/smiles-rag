from sklearn.cluster import KMeans
from baseline.knowledge_base import QdrantKnowledgeBase
import numpy as np
from .utils import (
    choose_instances_from_clusters,
    calculate_centroids,
    calculate_centroid_distance,
    EstimatedPoint,
)
import asyncio
import string
from baseline.config import settings
import copy
import openai
from qdrant_client import models


class Drafter:
    def __init__(self):
        self._knowledge_base = QdrantKnowledgeBase()

    def __call__(self, query: str, use_drafter: bool = True):
        """Get similar points with an estimation based on Lowe's score for the top 1 and the distance to the centroid of each cluster.
        If drafter is not used, the function returns only the top-1 point."""
        if use_drafter:
            retrieved_points, embedding = self._knowledge_base.get_similar_points(
                query, k_nearest=9
            )
            estimated_points, lowe_metric = self.estimate_points(
                points=retrieved_points
            )
            draft_answers = self.draft_answers(query, estimated_points)
        else:
            retrieved_points, embedding = self._knowledge_base.get_similar_points(
                query, k_nearest=1
            )
            estimated_points = EstimatedPoint(point=retrieved_points[0], distance=0.0)
            draft_answers = ""
            lowe_metric = 0

        return estimated_points, lowe_metric, draft_answers

    @staticmethod
    def draft_answers(query: str, points: list[EstimatedPoint]) -> list[str]:
        """Draft answers based on the selected points"""

        # Getting answers in parallel (async way). U can call this "костыль"
        async def get_answers() -> list[str]:
            nonlocal query
            nonlocal points

            client = openai.AsyncOpenAI(
                api_key=settings.drafter_api.key, base_url=settings.drafter_api.url
            )

            prompt = string.Template("something ... $query, $data")

            # Create async tasks
            tasks = []
            for point in points:
                data = point.point.payload["text"]
                prompt = prompt.substitute(query=query, data=data)
                task = client.chat.completions.create(
                    model="", messages=[{"role": "user", "content": prompt}]
                )
                tasks.append(task)
            # Run async tasks
            results = await asyncio.gather(*tasks)
            # Process outputs
            answers = [result.choices[0].message.content for result in results]
            return answers

        # run asyncio eventloop
        draft_answers = asyncio.run(get_answers())
        return draft_answers

    @staticmethod
    def estimate_points(
        points: list[models.ScoredPoint],
    ) -> tuple[list[EstimatedPoint], float]:
        """Estimate and select points by clustering and calculating distances to centroids and lowe's score"""
        vectors = np.array([point.vector for point in points])

        clusters = KMeans(n_clusters=3).fit_predict(vectors)

        # Get points from each cluster
        ind = choose_instances_from_clusters(clusters)
        selected_points = [points[i] for i in ind]
        selected_vectors = np.array([point.vector for point in selected_points])

        # Calculate distances to centroids
        centroids = calculate_centroids(clusters, vectors)
        distances_to_centroids_metric = calculate_centroid_distance(
            vectors=selected_vectors, centroids=centroids, use_softmax=True
        )

        # Calculate lowe's score
        lowe_metric = 1 - points[0].score / points[1].score

        # Collect points and their metrics
        results = []
        for point, distance in zip(selected_points, distances_to_centroids_metric):
            estimated_point = EstimatedPoint(point=point, distance=distance)
            results.append(copy.deepcopy(estimated_point))

        return results, lowe_metric
