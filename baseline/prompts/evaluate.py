system_prompt = "You are an answer evaluator."

user_prompt = """You are given the question, the correct answer, and the user's answer.
Your task is to compare the user's answer with the correct answer and determine if the user answered correctly.

QUESTION:
{0}

CORRECT ANSWER:
{1}

USER'S ANSWER:
{2}

Write "+" if the answer is correct, or "-" if the answer is incorrect. Never write anything other than "+" or "-".

EVALUATION:
"""