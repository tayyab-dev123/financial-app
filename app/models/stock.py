# app/models/stock.py
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
import pandas as pd


class StockData(BaseModel):
    """Representation of stock price data."""

    symbol: str
    data: pd.DataFrame
    timeframe: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    class Config:
        arbitrary_types_allowed = True


class NewsItem(BaseModel):
    """Representation of a news article."""

    id: str
    headline: str
    summary: Optional[str] = None
    url: str
    source: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    symbols: List[str] = []


class SentimentAnalysis(BaseModel):
    """Results of sentiment analysis on news."""

    overall_sentiment: str  # positive, negative, neutral
    sentiment_score: float  # 0 to 1, where 0 is negative, 1 is positive
    confidence: float  # 0 to 1
    summary: Optional[str] = None


class PriceTarget(BaseModel):
    """Price target with timeframe and confidence."""

    price: float
    timeframe: str  # Can be a specific timeframe or numbered target ("1", "2", etc.)
    confidence: float  # 0 to 1


class StockAnalysis(BaseModel):
    """Results of technical analysis on stock data."""

    symbol: str
    current_price: float
    prediction: str  # bullish, bearish, neutral
    confidence: float  # 0 to 1
    timeframe: str
    support_levels: List[float] = []
    resistance_levels: List[float] = []
    indicators: Dict[str, Optional[float]] = {}
    last_updated: datetime = Field(default_factory=datetime.now)


class TradingRecommendation(BaseModel):
    """Trading recommendation for a stock."""

    symbol: str
    action: str  # strong_buy, buy, hold, sell, strong_sell
    current_price: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: List[PriceTarget] = []
    confidence: float  # 0 to 1
    reasoning: Optional[str] = None
    timeframe: str  # days, weeks, months
    timestamp: datetime = Field(default_factory=datetime.now)


class UserQuery(BaseModel):
    """Representation of a parsed user query."""

    raw_query: str
    symbols: List[str]
    query_type: str  # recommendation, prediction, sentiment, technical, quote, general
    time_horizon: str  # short, medium, long
    risk_profile: str  # conservative, moderate, aggressive


class AssistantResponse(BaseModel):
    """Response from the financial assistant."""

    query: str
    response: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
