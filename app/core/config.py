# app/core/config.py
import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""

    # App info
    APP_NAME: str = "AI Financial Assistant"
    APP_VERSION: str = "0.1.0"

    # Security
    SECRET_KEY: str = Field(
        default_factory=lambda: os.environ.get("SECRET_KEY", "supersecretkey")
    )
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    PASSWORD_SALT: str = Field(
        default_factory=lambda: os.environ.get("PASSWORD_SALT", "some-salt")
    )

    # Alpaca API
    ALPACA_API_KEY: str = Field(
        default_factory=lambda: os.environ.get("ALPACA_API_KEY", "")
    )
    ALPACA_SECRET_KEY: str = Field(
        default_factory=lambda: os.environ.get("ALPACA_SECRET_KEY", "")
    )
    ALPACA_BASE_URL: str = Field(
        default_factory=lambda: os.environ.get(
            "ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2"
        )
    )

    # Database
    MONGODB_URL: str = Field(
        default_factory=lambda: os.environ.get(
            "MONGO_URI", os.environ.get("MONGODB_URL", "mongodb://localhost:27017")
        )
    )
    MONGODB_DATABASE: str = Field(
        default_factory=lambda: os.environ.get(
            "MONGO_DB_NAME", os.environ.get("MONGODB_DATABASE", "financial_assistant")
        )
    )

    # Add these fields to match env variables exactly
    MONGO_URI: str = ""
    MONGO_DB_NAME: str = ""

    # Huggingface
    HF_TOKEN: str = Field(default_factory=lambda: os.environ.get("HF_TOKEN", ""))

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",  # Allow extra fields from env variables
    }


# Create global settings object
settings = Settings()
