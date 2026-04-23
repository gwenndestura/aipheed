from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env")

    DATABASE_URL: str = "sqlite+aiosqlite:///./aipheed.db"
    SECRET_KEY: str = "changeme"
    GNEWS_API_KEY: str = ""
    MODEL_PATH: str = "models"
    PSA_MAIN_URL: str = "https://psa.gov.ph"
    PSA_RSSO_URL: str = "https://rsso04a.psa.gov.ph"
    DATA_RAW_PATH: str = "data/raw"


settings = Settings()