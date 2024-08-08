import openai
from baseline.config import settings
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
