from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


class QdrantConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")
    host: str
    port: int
    collection_name: str


class DrafterAPIConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")
    url: str | None = Field(alias="DRAFTER_URL", default=None)
    key: str = Field(alias="DRAFTER_KEY", default="")


class GeneratorAPIConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")
    url: str | None = Field(alias="GENERATOR_URL", default=None)
    key: str = Field(alias="GENERATOR_KEY", default="")


class Settings(BaseSettings):
    qdrant: QdrantConfig = QdrantConfig()
    drafter_api: DrafterAPIConfig = DrafterAPIConfig()


settings = Settings()
