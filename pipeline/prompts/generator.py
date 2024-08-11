system_prompt = "You are a smart assistant who answers questions based on 3 experts' answers, their explanations and these answers' metrics."

user_prompt = """QUESTION:
{0}

EXPERT ANSWERS:
Expert 1 (uncertainty {1}%): {2};
Expert 2 (uncertainty {3}%): {4};
Expert 3 (uncertainty {5}%): {6}.

Consider the expert opinions and the answers' metrics and provide an accurate and comprehensive answer to the question.

ANSWER:
"""

user_prompt_no_drafter = """You are a smart assistant. Answer the user's question using the provided information.

QUESTION:
{0}

INFORMATION:
{1}

ANSWER:
"""
