from typing import List

import openai

from baseline.config import settings
from baseline.prompts.llama import system_prompt, user_prompt, user_prompt_no_drafter
from baseline.rag.utils import EstimatedPoint

from .drafter import Drafter


class LMGenerator:
    def __init__(self):
        self._llm_client = openai.OpenAI(
            api_key=settings.llm_api.key,
            base_url=settings.llm_api.url
            if settings.llm_api.url is not None
            else "https://api.openai.com/v1",
        )
        self._drafter = Drafter()

    def __call__(
        self,
        query: str,
        use_drafter: bool = True,
    ) -> str:
        """Generate the final answer based on the selected points"""

        estimated_points, lowe_metric, draft_answers = self._drafter(
            query, create_answer=use_drafter
        )

        if use_drafter:
            answer = self._answer_with_drafter(
                query, estimated_points, lowe_metric, draft_answers
            )
            return answer

        else:
            answer = self._answer_without_drafter(query, estimated_points)
            return answer

    def _answer_with_drafter(
        self,
        query: str,
        estimated_points: List[EstimatedPoint],
        lowe_metric: float,
        draft_answers: List[str],
    ) -> str:
        prompt = user_prompt.format(
            query,
            lowe_metric,
            estimated_points[0].distance,
            draft_answers[0],
            estimated_points[1].distance,
            draft_answers[1],
            estimated_points[2].distance,
            draft_answers[2],
        )

        result = self._llm_client.chat.completions.create(
            model="neuralmagic/gemma-2-2b-it-FP8",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            top_p=50,
            max_tokens=500,
        )

        answer = result.choices[0].message
        return answer

    def _answer_without_drafter(
        self, query: str, estimated_points: List[EstimatedPoint]
    ) -> str:
        text = estimated_points[0].point.payload["text"]

        prompt = user_prompt_no_drafter.format(query, text)

        result = self._llm_client.chat.completions.create(
            model="neuralmagic/gemma-2-2b-it-FP8",
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            top_p=50,
            max_tokens=500,
        )

        answer = result.choices[0].message
        return answer
