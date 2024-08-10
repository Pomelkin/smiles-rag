from sklearn.cluster import KMeans
from baseline.knowledge_base import QdrantKnowledgeBase
import numpy as np
from .utils import (
    choose_instances_from_clusters,
    calculate_centroids,
    calculate_uncertainty,
    EstimatedPoint,
    calculate_preference,
)
from baseline.config import settings
import copy

import openai
from concurrent.futures import ThreadPoolExecutor
import concurrent
from qdrant_client import models

from baseline.prompts.gemma import user_prompt


class Drafter:
    def __init__(self):
        self._knowledge_base = QdrantKnowledgeBase()
        self._client = openai.OpenAI(
            api_key=settings.drafter_api.key,
            base_url=settings.drafter_api.url,
        )

    def __call__(self, query: str, create_answer: bool = True):
        """Get similar points with an estimation based on Lowe's score for the top 1 and top 2 similarity score
        and the distance to the centroid of each cluster.
        If drafter is not used, the function returns only the top-1 point."""
        if create_answer:
            retrieved_points, embedding = self._knowledge_base.get_similar_points(
                query, k_nearest=9
            )
            estimated_points, preference_metric = self.estimate_points(
                points=retrieved_points
            )
            draft_answers = self.draft_answers(query, estimated_points)
        else:
            retrieved_points, embedding = self._knowledge_base.get_similar_points(
                query, k_nearest=1
            )
            estimated_points = [].append(
                EstimatedPoint(point=retrieved_points[0], uncertainty=0.0)
            )
            draft_answers = ""
            preference_metric = 0
        return estimated_points, preference_metric, draft_answers

    def draft_answers(self, query: str, points: list[EstimatedPoint]) -> list[str]:
        """Draft answers based on the selected points"""
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Create async tasks
            futures = []
            for i, point in enumerate(points):
                data = point.point.payload["text"]
                prompt = user_prompt.format(query, data)

                future = executor.submit(self.generate_answer, i, prompt)
                futures.append(future)

            # Run async tasks
            done, _ = concurrent.futures.wait(futures)

            # Postprocess results
            results = ["" for _ in range(len(futures))]
            for future in done:
                results[future.result()[0]] = future.result()[1]
            answers = results
        return answers

    def generate_answer(self, ind: int, prompt: str) -> tuple[int, str]:
        """Generate the final answer based on prompt.
        Ind is needed for sorting purposes due to the behavior of the ThreadPoolExecutor wait method"""
        result = self._client.chat.completions.create(
            model="neuralmagic/gemma-2-2b-it-FP8",
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            top_p=0.8,
            max_tokens=500,
        )
        answer = result.choices[0].message.content
        return ind, answer

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
        uncertainty_metric = calculate_uncertainty(
            vectors=selected_vectors, centroids=centroids, use_softmax=True
        )

        # Calculate lowe's score
        preference_metric = calculate_preference(
            top_1_score=points[0].score, top_2_score=points[1].score
        )

        # Collect points and their metrics
        results = []
        for point, uncertainty in zip(selected_points, uncertainty_metric):
            estimated_point = EstimatedPoint(point=point, uncertainty=uncertainty)
            results.append(copy.deepcopy(estimated_point))

        return results, preference_metric
