# app/agents/analysis_agent.py
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.services.huggingface_service import HuggingFaceService
from app.models.stock import StockData, StockAnalysis, SentimentAnalysis


class AnalysisAgent:
    """Agent responsible for analyzing stock data and news sentiment."""

    def __init__(self):
        self.hf_service = HuggingFaceService()

    async def analyze_stock_trend(self, stock_data: StockData) -> StockAnalysis:
        """
        Analyze stock price trends using ML models.

        Args:
            stock_data: Historical stock data

        Returns:
            StockAnalysis object with trend predictions and indicators
        """
        df = stock_data.data

        # Calculate technical indicators
        df = self._calculate_technical_indicators(df)

        # Prepare features for prediction
        features = self._prepare_prediction_features(df)

        # Make predictions using the trend prediction model
        predictions = await self.hf_service.predict_trend(features)

        # Calculate price targets
        current_price = df["close"].iloc[-1]
        support, resistance = self._calculate_support_resistance(df)

        return StockAnalysis(
            symbol=stock_data.symbol,
            current_price=current_price,
            prediction=predictions["prediction"],
            confidence=predictions["confidence"],
            timeframe=stock_data.timeframe,
            support_levels=[support],
            resistance_levels=[resistance],
            indicators={
                "rsi": df["rsi"].iloc[-1] if "rsi" in df else None,
                "macd": df["macd"].iloc[-1] if "macd" in df else None,
                "macd_signal": (
                    df["macd_signal"].iloc[-1] if "macd_signal" in df else None
                ),
                "ma_50": df["ma_50"].iloc[-1] if "ma_50" in df else None,
                "ma_200": df["ma_200"].iloc[-1] if "ma_200" in df else None,
            },
            last_updated=datetime.now(),
        )

    async def analyze_news_sentiment(self, news_items: List[Dict]) -> SentimentAnalysis:
        """
        Analyze sentiment of news articles related to a stock.

        Args:
            news_items: List of news articles

        Returns:
            SentimentAnalysis object with sentiment scores and summary
        """
        if not news_items:
            return SentimentAnalysis(
                overall_sentiment="neutral",
                sentiment_score=0.5,
                confidence=0.0,
                summary="No news articles available for analysis.",
            )

        # Extract text content from news items
        texts = [
            item.get("headline", "") + ". " + item.get("summary", "")
            for item in news_items
            if item.get("headline")
        ]

        # Get sentiment analysis from HuggingFace model
        sentiments = await self.hf_service.analyze_sentiment(texts)

        # Calculate overall sentiment
        if not sentiments:
            return SentimentAnalysis(
                overall_sentiment="neutral",
                sentiment_score=0.5,
                confidence=0.0,
                summary="Failed to analyze sentiment of news articles.",
            )

        # Average the sentiment scores
        avg_score = sum(s["score"] for s in sentiments) / len(sentiments)
        avg_confidence = sum(s["confidence"] for s in sentiments) / len(sentiments)

        # Determine overall sentiment label
        if avg_score > 0.6:
            sentiment_label = "positive"
        elif avg_score < 0.4:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"

        # Generate a summary of the sentiment analysis
        summary = (
            f"Analysis of {len(texts)} news articles shows {sentiment_label} sentiment "
        )
        summary += f"with an average score of {avg_score:.2f} (confidence: {avg_confidence:.2f})."

        return SentimentAnalysis(
            overall_sentiment=sentiment_label,
            sentiment_score=avg_score,
            confidence=avg_confidence,
            summary=summary,
        )

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for stock analysis."""
        # Make a copy to avoid modifying the original
        df = df.copy()

        # Calculate moving averages
        df["ma_50"] = df["close"].rolling(window=50).mean()
        df["ma_200"] = df["close"].rolling(window=200).mean()

        # Calculate RSI
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)  # Avoid division by zero
        df["rsi"] = 100 - (100 / (1 + rs))

        # Calculate MACD
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

        return df

    def _prepare_prediction_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare features for the prediction model."""
        # Get the last few days of data with indicators
        last_rows = df.tail(10).copy()

        # Fill any missing values
        last_rows = last_rows.fillna(method="ffill").fillna(method="bfill")

        # Extract features
        features = {
            "prices": last_rows["close"].values.tolist(),
            "volumes": last_rows["volume"].values.tolist(),
            "rsi": last_rows["rsi"].values.tolist() if "rsi" in last_rows else [],
            "macd": last_rows["macd"].values.tolist() if "macd" in last_rows else [],
            "macd_signal": (
                last_rows["macd_signal"].values.tolist()
                if "macd_signal" in last_rows
                else []
            ),
        }

        return features

    def _calculate_support_resistance(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate basic support and resistance levels."""
        if len(df) < 20:
            return df["close"].min(), df["close"].max()

        # Get recent price data
        recent_data = df.tail(20)

        # Simple method: Use recent lows and highs
        support = recent_data["low"].min()
        resistance = recent_data["high"].max()

        return support, resistance
