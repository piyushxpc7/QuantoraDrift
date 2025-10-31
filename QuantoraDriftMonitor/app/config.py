from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    database_url: str
    app_name: str = "Quantora - Quantitative Model Monitoring System"
    app_version: str = "1.0.0"
    debug: bool = False
    
    psi_threshold: float = 0.25
    ks_threshold: float = 0.2
    sharpe_window: int = 30
    sharpe_threshold: float = -0.5
    
    min_samples_for_analysis: int = 20
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings():
    return Settings()
