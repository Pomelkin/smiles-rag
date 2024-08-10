from typing import List

import openai
from pipeline.rag.drafter import Drafter

from pipeline.config import settings
from pipeline.prompts.generator import (
    system_prompt,
    user_prompt,
    user_prompt_no_drafter,
)
from pipeline.rag.utils import EstimatedPoint


class LMGenerator:
    def __init__(self):
        self._llm_client = openai.OpenAI(
            api_key=settings.generator_api.key,
            base_url=settings.generator_api.url
            if settings.generator_api.url is not None
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
        # Convert numbers to percentages for the prompt
        prompt = user_prompt.format(
            query,
            lowe_metric * 100,
            estimated_points[0].uncertainty * 100,
            draft_answers[0],
            estimated_points[1].uncertainty * 100,
            draft_answers[1],
            estimated_points[2].uncertainty * 100,
            draft_answers[2],
        )

        result = self._llm_client.chat.completions.create(
            model="neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            top_p=50,
            max_tokens=500,
        )

        answer = result.choices[0].message.content
        return answer

    def _answer_without_drafter(
        self, query: str, estimated_points: List[EstimatedPoint]
    ) -> str:
        text = estimated_points[0].point.payload["text"]

        prompt = user_prompt_no_drafter.format(query, text)

        result = self._llm_client.chat.completions.create(
            model="neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16",
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            top_p=50,
            max_tokens=500,
        )

        answer = result.choices[0].message.content
        return answer
