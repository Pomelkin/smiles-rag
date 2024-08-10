from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


class QdrantConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")
    host: str
    port: int
    collection_name: str


class RedisConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")
    host: str = Field(alias="REDIS_HOST", default="localhost")
    port: int = Field(alias="REDIS_PORT", default=6379)


class DrafterAPIConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")
    url: str = Field(alias="DRAFTER_URL", default=None, validate_default=False)
    key: str = Field(alias="DRAFTER_KEY", default="")


class GeneratorAPIConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")
    url: str = Field(alias="GENERATOR_URL", default=None, validate_default=False)
    key: str = Field(alias="GENERATOR_KEY", default="")


class Settings(BaseSettings):
    qdrant: QdrantConfig = QdrantConfig()
    drafter_api: DrafterAPIConfig = DrafterAPIConfig()
    generator_api: GeneratorAPIConfig = GeneratorAPIConfig()
    redis: RedisConfig = RedisConfig()


settings = Settings()
