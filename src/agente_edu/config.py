from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    soberano_api_base_url: str = "https://api.sobdemanda.mandu.piaui.pro/v1"
    soberano_api_key: str = ""
    soberano_model: str = "Qwen/Qwen3.6-35B-A3B"

    embeddings_provider: str = "local"
    embeddings_model: str = "BAAI/bge-m3"

    data_path: str = "data/pib_piaui.csv"
    vector_store: str = "faiss"
    vector_store_path: str = ".index"

    default_persona: str = "ensino_medio"
    top_k: int = 5

    interface: str = "web"
    telegram_bot_token: str = ""
    web_port: int = 8000

    redis_url: str = ""
    rate_limit_por_min: int = 10
    log_level: str = "INFO"

    default_base_id: str = "pib"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()

# Quando instalado via pip, __file__ aponta para site-packages.
# Usamos APP_BASE_DIR (env) com fallback para o diretório de trabalho atual,
# que é a raiz do projeto tanto localmente quanto no Docker (WORKDIR /app).
import os as _os
BASE_DIR = Path(_os.environ.get("APP_BASE_DIR", Path.cwd()))
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / settings.vector_store_path
