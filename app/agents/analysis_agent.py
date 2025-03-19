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
        try:
            df = stock_data.data

            if df.empty:
                # Handle empty dataframe case
                return StockAnalysis(
                    symbol=stock_data.symbol,
                    current_price=0.0,
                    prediction="neutral",
                    confidence=0.0,
                    timeframe=stock_data.timeframe,
                    support_levels=[0.0],
                    resistance_levels=[0.0],
                    indicators={
                        "rsi": None,
                        "macd": None,
                        "macd_signal": None,
                        "ma_50": None,
                        "ma_200": None,
                    },
                    last_updated=datetime.now(),
                )

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
        except Exception as e:
            print(f"Error in analyze_stock_trend: {str(e)}")
            # Return a default analysis with error information
            return StockAnalysis(
                symbol=stock_data.symbol,
                current_price=0.0,
                prediction="neutral",
                confidence=0.0,
                timeframe=stock_data.timeframe,
                support_levels=[0.0],
                resistance_levels=[0.0],
                indicators={},
                last_updated=datetime.now(),
                error=str(e),
            )

    async def analyze_news_sentiment(self, news_items: List[Dict]) -> SentimentAnalysis:
        """
        Analyze sentiment of news articles related to a stock.

        Args:
            news_items: List of news articles (can be dictionaries or NewsItem objects)

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

        # Extract text content from news items, handling both dict and NewsItem objects
        texts = []
        for item in news_items:
            # Handle both dictionary and NewsItem objects
            if hasattr(item, "dict") and callable(getattr(item, "dict")):
                # It's a Pydantic model (NewsItem)
                item_dict = item.dict()
                headline = item_dict.get("headline", "")
                summary = item_dict.get("summary", "")
            else:
                # It's already a dictionary
                headline = item.get("headline", "")
                summary = item.get("summary", "")

            if headline:
                texts.append(f"{headline}. {summary}")

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
        try:
            # Make a copy to avoid modifying the original
            df = df.copy()

            # Make sure we have numeric data
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Ensure enough data for calculations
            if len(df) < 50:
                # Return dataframe with NaN indicators if not enough data
                df["ma_50"] = np.nan
                df["ma_200"] = np.nan
                df["rsi"] = np.nan
                df["macd"] = np.nan
                df["macd_signal"] = np.nan
                return df

            # Calculate moving averages
            df["ma_50"] = df["close"].rolling(window=min(50, len(df))).mean()
            df["ma_200"] = df["close"].rolling(window=min(200, len(df))).mean()

            # Calculate RSI
            delta = df["close"].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=min(14, len(df))).mean()
            avg_loss = loss.rolling(window=min(14, len(df))).mean()
            rs = avg_gain / avg_loss.replace(0, 1e-10)  # Avoid division by zero
            df["rsi"] = 100 - (100 / (1 + rs))

            # Calculate MACD using vectorized operations
            ema_12 = df["close"].ewm(span=min(12, len(df)), adjust=False).mean()
            ema_26 = df["close"].ewm(span=min(26, len(df)), adjust=False).mean()
            df["macd"] = ema_12 - ema_26
            df["macd_signal"] = (
                df["macd"].ewm(span=min(9, len(df)), adjust=False).mean()
            )

            return df
        except Exception as e:
            print(f"Error calculating technical indicators: {str(e)}")
            # Return the original dataframe if there's an error
            return df

    def _prepare_prediction_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare features for the prediction model."""
        try:
            # Get the last few days of data with indicators
            rows_to_use = min(10, len(df))
            last_rows = df.tail(rows_to_use).copy()

            # Fill any missing values - using recommended methods instead of deprecated ones
            last_rows = last_rows.ffill().bfill()

            # Safely extract features
            features = {
                "prices": (
                    last_rows["close"].astype(float).tolist()
                    if "close" in last_rows
                    else []
                ),
                "volumes": (
                    last_rows["volume"].astype(float).tolist()
                    if "volume" in last_rows
                    else []
                ),
                "rsi": (
                    last_rows["rsi"].astype(float).tolist()
                    if "rsi" in last_rows and not last_rows["rsi"].isna().all()
                    else []
                ),
                "macd": (
                    last_rows["macd"].astype(float).tolist()
                    if "macd" in last_rows and not last_rows["macd"].isna().all()
                    else []
                ),
                "macd_signal": (
                    last_rows["macd_signal"].astype(float).tolist()
                    if "macd_signal" in last_rows
                    and not last_rows["macd_signal"].isna().all()
                    else []
                ),
            }

            return features
        except Exception as e:
            print(f"Error preparing prediction features: {str(e)}")
            # Return minimal features if there's an error
            return {
                "prices": [],
                "volumes": [],
                "rsi": [],
                "macd": [],
                "macd_signal": [],
            }

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
