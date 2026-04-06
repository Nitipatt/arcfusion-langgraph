from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    """LangGraph Service configuration."""

    # LLM Configuration
    LLM_PROVIDER: Literal["openai", "anthropic", "azure"] = "openai"
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_API_VERSION: str = ""
    LLM_MODEL: str = ""

    # Database (PostgreSQL) - for schema extraction and query execution
    DB_HOST: str = "localhost"
    DB_PORT: int = 5555
    DB_NAME: str = "arcfusion"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "12345"
    DB_SCHEMA: str = "public"
    DB_SSLMODE: str = "disable"

    # Service
    SERVICE_PORT: int = 9002
    DEBUG: bool = False
    ALLOWED_ORIGINS: str = "http://localhost:9001"

    # Cache Configuration
    CACHE_ENABLED: bool = True
    CACHE_SIMILARITY_THRESHOLD: float = 0.75
    CACHE_TTL_SECONDS: int = 3600        # 1 hour
    CACHE_MAX_SIZE: int = 500
    SCHEMA_CACHE_TTL_SECONDS: int = 1800  # 30 minutes

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
            f"?options=-csearch_path%3D{self.DB_SCHEMA}&sslmode={self.DB_SSLMODE}"
        )

    @property
    def effective_model(self) -> str:
        if self.LLM_MODEL:
            return self.LLM_MODEL
        if self.LLM_PROVIDER == "anthropic":
            return "claude-sonnet-4-20250514"
        return "gpt-4o"

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()
