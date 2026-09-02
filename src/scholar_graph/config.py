from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SCHOLAR_GRAPH_", env_file=".env", extra="ignore")

    openalex_email: str | None = None
    max_concurrent_requests: int = 5

settings = Settings()
