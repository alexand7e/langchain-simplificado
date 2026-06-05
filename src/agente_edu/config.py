from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    soberano_api_base_url: str = "https://<endpoint-da-mandu>"
    soberano_api_key: str = ""
    soberano_model: str = "soberano-1.1"

    embeddings_provider: str = "local"
    embeddings_model: str = "intfloat/multilingual-e5-small"

    data_path: str = "data/pib_piaui.csv"
    vector_store: str = "faiss"
    vector_store_path: str = ".index"

    default_persona: str = "ensino_medio"
    top_k: int = 3

    interface: str = "web"
    telegram_bot_token: str = ""
    web_port: int = 8000

    redis_url: str = ""
    rate_limit_por_min: int = 10
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / settings.vector_store_path
