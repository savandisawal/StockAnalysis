"""Application configuration loaded from environment variables."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Claude API
    anthropic_api_key: str = ""

    # Auth — comma-separated "user:pass" pairs
    auth_users: str = "admin:changeme"

    # Database
    database_url: str = f"sqlite+aiosqlite:///{PROJECT_ROOT / 'stock_quant.db'}"

    # News API (optional fallback)
    news_api_key: str = ""

    # Logging
    log_level: str = "INFO"

    # Scheduler — cron for weekly retraining (Friday 4 PM IST)
    retrain_cron: str = "0 16 * * 5"

    # Data fetch settings
    yfinance_max_retries: int = 3
    yfinance_retry_delay: float = 2.0
    ohlc_history_years: int = 3

    # Cache TTL in seconds
    cache_ttl_ohlc: int = 3600  # 1 hour
    cache_ttl_macro: int = 1800  # 30 min
    cache_ttl_sentiment: int = 14400  # 4 hours
    cache_ttl_fundamentals: int = 86400  # 24 hours

    # Model
    model_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "model" / "saved_models")

    # Sentiment
    sentiment_model: str = "claude-haiku-4-5-20251001"
    sentiment_max_headlines: int = 10

    def get_auth_credentials(self) -> dict[str, str]:
        """Parse AUTH_USERS into {username: password} dict."""
        creds = {}
        for pair in self.auth_users.split(","):
            pair = pair.strip()
            if ":" in pair:
                user, pwd = pair.split(":", 1)
                creds[user.strip()] = pwd.strip()
        return creds


settings = Settings()
