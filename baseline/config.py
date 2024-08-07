from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


class QdrantConfig(BaseModel):
    host: str
    port: int


class Settings(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True)

    qdrant: QdrantConfig


settings = Settings()
