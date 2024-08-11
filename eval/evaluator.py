import logging
import os

import openai
import pandas as pd

from eval.parser import DataParser
from pipeline.config import settings
from pipeline.prompts.evaluate import system_prompt, user_prompt
from pipeline.rag.generator import LMGenerator

logging.basicConfig(filename="logs.log", level=logging.INFO)


class Evaluator:
    def __init__(self):
        self._llm_client = openai.OpenAI(
            api_key=settings.generator_api.key,
            base_url=settings.generator_api.url
            if settings.generator_api.url is not None
            else "https://api.openai.com/v1",
        )

        self._parser = DataParser(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "qa")
        )

    def _get_llm_evaluation(self, query, answer, truth):
        prompt = user_prompt.format(query, truth, answer)

        result = self._llm_client.chat.completions.create(
            model="neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=2,
        )
        answer = result.choices[0].message.content.strip()

        return answer

    @staticmethod
    def _calculate_accuracy(df):
        total_count = len(df)
        plus_count = df["evaluation"].value_counts().get("+", 0)
        accuracy = (plus_count / total_count) * 100 if total_count > 0 else 0
        return accuracy

    def iterate(self):
        pipe_df = pd.DataFrame(columns=["question", "answer", "truth", "evaluation"])

        baseline_df = pd.DataFrame(
            columns=["question", "answer", "truth", "evaluation"]
        )

        pipe = LMGenerator()

        logging.info("Starting evaluation...")

        for j in range(len(self._parser)):
            parser = self._parser[j]

            for i, data in enumerate(parser):
                query = data["question"]
                truth = data["gt_answer"]

                pipe_answer = pipe(query, use_drafter=True)
                baseline_answer = pipe(query, use_drafter=False)

                pipe_evaluation = self._get_llm_evaluation(query, pipe_answer, truth)
                baseline_evaluation = self._get_llm_evaluation(
                    query, baseline_answer, truth
                )

                new_row = pd.DataFrame(
                    {
                        "question": [query],
                        "answer": [pipe_answer],
                        "truth": [truth],
                        "evaluation": [pipe_evaluation],
                    }
                )

                # Concatenate the DataFrames
                pipe_df = pd.concat([pipe_df, new_row], ignore_index=True)

                baseline_df = pd.concat(
                    [
                        baseline_df,
                        pd.DataFrame(
                            {
                                "question": [query],
                                "answer": [baseline_answer],
                                "truth": [truth],
                                "evaluation": [baseline_evaluation],
                            }
                        ),
                    ],
                    ignore_index=True,
                )

                if i % 10 == 0:
                    pipe_accuracy = self._calculate_accuracy(pipe_df)
                    baseline_accuracy = self._calculate_accuracy(baseline_df)
                    logging.info(
                        f"Pipe accuracy: {pipe_accuracy:.2f}%, Baseline accuracy: {baseline_accuracy:.2f}%"
                    )

        pipe_accuracy = self._calculate_accuracy(pipe_df)
        baseline_accuracy = self._calculate_accuracy(baseline_df)
        logging.info(
            f"Pipe accuracy: {pipe_accuracy:.2f}%, Baseline accuracy: {baseline_accuracy:.2f}%"
        )

        pipe_df.to_csv("pipe_eval.csv", index=False)
        baseline_df.to_csv("baseline_eval.csv", index=False)


if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.iterate()
