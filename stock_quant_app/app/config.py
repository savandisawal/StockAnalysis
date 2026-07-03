"""Application configuration loaded from environment variables.

Supports both local .env files and Streamlit Cloud secrets.
"""

import os
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_streamlit_secrets():
    """Inject Streamlit Cloud secrets into environment variables."""
    try:
        import streamlit as st

        for key, val in st.secrets.items():
            if isinstance(val, str) and key not in os.environ:
                os.environ[key] = val
    except Exception:
        pass


_load_streamlit_secrets()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Base directory for all mutable state (DBs, logs, models).
    # Defaults to the project root locally; set DATA_DIR=/data in Docker.
    data_dir: Path = PROJECT_ROOT

    # Deployment environment: "dev" or "prod" (prod enforces hashed auth)
    environment: str = "dev"

    # Claude API
    anthropic_api_key: SecretStr = SecretStr("")

    # Auth — comma-separated "user:bcrypt_hash" pairs. Empty = all requests
    # rejected (fail closed). Generate hashes with: python -m scripts.hash_password
    auth_users: str = ""

    # Database — derived from data_dir when left empty
    database_url: str = ""

    # News API (optional fallback)
    news_api_key: SecretStr = SecretStr("")

    # Logging
    log_level: str = "INFO"
    log_json: bool = True  # JSON-lines file sink (logs/app.jsonl)

    # API hardening
    cors_origins: str = ""  # comma-separated allowed origins; empty = same-origin only
    rate_limit_default: str = "60/minute"
    rate_limit_heavy: str = "3/minute"  # train / backtest

    # Scheduler — cron for weekly retraining (Friday 4 PM IST)
    retrain_cron: str = "0 16 * * 5"

    # Stocks auto-refreshed, retrained, and predicted daily by the scheduler
    tracked_stocks: str = (
        "RELIANCE,TCS,HDFCBANK,INFY,ICICIBANK,KOTAKBANK,SBIN,BHARTIARTL,ITC,TATAMOTORS"
    )

    # Data fetch settings
    yfinance_max_retries: int = 3
    yfinance_retry_delay: float = 2.0
    ohlc_history_years: int = 3

    # Cache TTL in seconds
    cache_ttl_ohlc: int = 3600  # 1 hour
    cache_ttl_macro: int = 1800  # 30 min
    cache_ttl_sentiment: int = 14400  # 4 hours
    cache_ttl_fundamentals: int = 86400  # 24 hours
    cache_ttl_fundamental_history: int = 604800  # 7 days — quarterly data moves slowly

    # Model — derived from data_dir when left unset
    model_dir: Path | None = Field(default=None)

    # Sentiment
    sentiment_model: str = "claude-haiku-4-5-20251001"
    sentiment_max_headlines: int = 10

    def model_post_init(self, __context) -> None:
        if not self.database_url:
            self.database_url = f"sqlite+aiosqlite:///{self.db_path}"
        if self.model_dir is None:
            self.model_dir = self.data_dir / "model" / "saved_models"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "stock_quant.db"

    @property
    def cache_db_path(self) -> Path:
        return self.data_dir / "cache.db"

    @property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"

    def get_auth_credentials(self) -> dict[str, str]:
        """Parse AUTH_USERS into {username: bcrypt_hash} dict.

        bcrypt hashes never contain ':', so split(":", 1) is safe.
        """
        creds = {}
        for pair in self.auth_users.split(","):
            pair = pair.strip()
            if ":" in pair:
                user, pwd_hash = pair.split(":", 1)
                creds[user.strip()] = pwd_hash.strip()
        return creds

    def get_cors_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
