system_prompt = "You are a smart assistant who answers questions based on 3 experts' answers and these answers' metrics. The answers are sorted by experts' rating."

user_prompt = """QUESTION:
{0}

EXPERT ANSWERS:
Expert 1 (preference {1}%, uncertainty {2}%): {3};
Expert 2 (uncertainty {4}%): {5};
Expert 3 (uncertainty {6}%): {7}.

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
