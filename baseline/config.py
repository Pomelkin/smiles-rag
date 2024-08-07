from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


class QdrantConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env")
    host: str
    port: int
    collection_name: str


class Settings(BaseSettings):
    qdrant: QdrantConfig = QdrantConfig()


settings = Settings()
