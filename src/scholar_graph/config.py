from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openalex_email: str | None = None
    max_concurrent_requests: int = 5
    
    class Config:
        env_prefix = "SCHOLAR_GRAPH_"
        env_file = ".env"

settings = Settings()
