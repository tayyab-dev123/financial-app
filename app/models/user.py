# app/models/user.py
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, EmailStr


class UserPreferences(BaseModel):
    """User preferences for the financial assistant."""

    risk_profile: str = "moderate"  # conservative, moderate, aggressive
    default_investment_horizon: str = "medium"  # short, medium, long
    favorite_symbols: List[str] = []
    notification_preferences: Dict[str, bool] = Field(
        default_factory=lambda: {
            "price_alerts": True,
            "trading_signals": True,
            "news_alerts": True,
        }
    )


class User(BaseModel):
    """User profile information."""

    id: str
    username: str
    email: EmailStr
    created_at: datetime = Field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    preferences: UserPreferences = Field(default_factory=UserPreferences)


class UserPortfolio(BaseModel):
    """User's investment portfolio."""

    user_id: str
    holdings: Dict[str, float] = {}  # Symbol to quantity mapping
    cash_balance: float = 0.0
    last_updated: datetime = Field(default_factory=datetime.now)


class UserAlert(BaseModel):
    """Price or event alert for a user."""

    id: str
    user_id: str
    symbol: str
    alert_type: str  # price_target, price_change, technical_signal, news
    condition: Dict[str, Any]  # Condition details specific to alert type
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    triggered_at: Optional[datetime] = None


class UserSession(BaseModel):
    """User session data."""

    id: str
    user_id: str
    started_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    conversation_history: List[Dict[str, Any]] = []
