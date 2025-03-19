# app/services/huggingface_service.py
from typing import Dict, List, Any, Optional
import os
import numpy as np
from transformers import pipeline
import torch
import asyncio
from functools import lru_cache


class HuggingFaceService:
    """Service to interact with HuggingFace Transformers models."""

    def __init__(self):
        # Initialize the models as needed
        self._initialize_models()

    @lru_cache(maxsize=1)
    def _initialize_models(self):
        """Initialize and cache the transformer models."""
        # Use CUDA if available
        device = 0 if torch.cuda.is_available() else -1

        # Initialize sentiment analysis model
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device,
        )

        # Initialize text classification model for financial sentiment
        self.financial_sentiment_pipeline = pipeline(
            "text-classification", model="ProsusAI/finbert", device=device
        )

        # Initialize time series forecasting model
        # Note: Using a custom model for stock prediction would be better,
        # but we'll use a text-to-text model as a placeholder
        self.forecasting_pipeline = pipeline(
            "text2text-generation", model="google-t5/t5-small", device=device
        )

    async def analyze_sentiment(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment of text using transformer models.

        Args:
            texts: List of text strings to analyze

        Returns:
            List of sentiment analysis results with scores and labels
        """
        if not texts:
            return []

        # Run sentiment analysis in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, lambda: self.financial_sentiment_pipeline(texts)
        )

        # Process and normalize results
        processed_results = []
        for idx, result in enumerate(results):
            label = result["label"].lower()

            # Convert finbert labels to scores between 0 and 1
            if label == "positive":
                score = 0.75 + (result["score"] * 0.25)  # Range: 0.75-1.0
            elif label == "negative":
                score = 0.25 - (result["score"] * 0.25)  # Range: 0-0.25
            else:  # neutral
                score = 0.5  # Middle value

            processed_results.append(
                {
                    "text": (
                        texts[idx][:100] + "..."
                        if len(texts[idx]) > 100
                        else texts[idx]
                    ),
                    "sentiment": label,
                    "score": score,
                    "confidence": result["score"],
                }
            )

        return processed_results

    async def predict_trend(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict stock price trend using ML models.

        Args:
            features: Dict containing price history and technical indicators

        Returns:
            Dict with prediction results (direction, confidence, etc.)
        """
        # In a production system, you'd use a proper time series model
        # This is a simplified placeholder that uses basic trend analysis

        # Extract recent prices
        prices = features.get("prices", [])
        if not prices or len(prices) < 2:
            return {
                "prediction": "neutral",
                "confidence": 0.5,
                "price_target": None,
                "timeframe": "1D",
            }

        # Simple trend analysis based on recent price movements
        recent_prices = prices[-5:]  # Last 5 data points
        price_changes = [
            recent_prices[i] - recent_prices[i - 1]
            for i in range(1, len(recent_prices))
        ]
        avg_change = sum(price_changes) / len(price_changes)

        # Get momentum from RSI if available
        rsi_values = features.get("rsi", [])
        latest_rsi = rsi_values[-1] if rsi_values else 50

        # Get MACD signals if available
        macd_values = features.get("macd", [])
        macd_signal_values = features.get("macd_signal", [])

        macd_signal = 0
        if (
            macd_values
            and macd_signal_values
            and len(macd_values) > 1
            and len(macd_signal_values) > 1
        ):
            # Check for MACD crossover
            if (
                macd_values[-2] < macd_signal_values[-2]
                and macd_values[-1] >= macd_signal_values[-1]
            ):
                macd_signal = 1  # Bullish crossover
            elif (
                macd_values[-2] > macd_signal_values[-2]
                and macd_values[-1] <= macd_signal_values[-1]
            ):
                macd_signal = -1  # Bearish crossover

        # Combine signals
        signals = []

        # Price trend signal
        if avg_change > 0:
            signals.append(
                ("price_trend", "bullish", min(abs(avg_change) / prices[-1] * 100, 1.0))
            )
        else:
            signals.append(
                ("price_trend", "bearish", min(abs(avg_change) / prices[-1] * 100, 1.0))
            )

        # RSI signal
        if latest_rsi > 70:
            signals.append(("rsi", "bearish", (latest_rsi - 70) / 30))
        elif latest_rsi < 30:
            signals.append(("rsi", "bullish", (30 - latest_rsi) / 30))
        else:
            signals.append(("rsi", "neutral", 0.5))

        # MACD signal
        if macd_signal == 1:
            signals.append(("macd", "bullish", 0.8))
        elif macd_signal == -1:
            signals.append(("macd", "bearish", 0.8))
        else:
            signals.append(("macd", "neutral", 0.5))

        # Combine all signals with their confidence
        bullish_confidence = sum(
            conf for _, direction, conf in signals if direction == "bullish"
        )
        bearish_confidence = sum(
            conf for _, direction, conf in signals if direction == "bearish"
        )

        # Normalize and determine final prediction
        total_signals = len(signals)
        bullish_score = bullish_confidence / total_signals
        bearish_score = bearish_confidence / total_signals

        if bullish_score > bearish_score:
            prediction = "bullish"
            confidence = bullish_score
        elif bearish_score > bullish_score:
            prediction = "bearish"
            confidence = bearish_score
        else:
            prediction = "neutral"
            confidence = 0.5

        # Calculate price target (very simplified)
        current_price = prices[-1]
        price_target = None
        if prediction == "bullish":
            price_target = current_price * (
                1 + 0.02 * confidence
            )  # 2% target scaled by confidence
        elif prediction == "bearish":
            price_target = current_price * (
                1 - 0.02 * confidence
            )  # 2% target scaled by confidence

        return {
            "prediction": prediction,
            "confidence": confidence,
            "price_target": price_target,
            "timeframe": "1D",
            "signals": {
                signal[0]: {"direction": signal[1], "strength": signal[2]}
                for signal in signals
            },
        }

    async def generate_summary(
        self, text_list: List[str], max_length: int = 150
    ) -> str:
        """
        Generate a concise summary of multiple text inputs.

        Args:
            text_list: List of text strings to summarize
            max_length: Maximum length of the summary

        Returns:
            Summarized text
        """
        if not text_list:
            return ""

        # Combine texts (limit to avoid context length issues)
        combined_text = " ".join(text_list)
        if len(combined_text) > 1000:
            combined_text = combined_text[:1000] + "..."

        prompt = f"summarize: {combined_text}"

        # Run text generation in a thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.forecasting_pipeline(
                prompt, max_length=max_length, num_return_sequences=1
            ),
        )

        if result and isinstance(result, list) and len(result) > 0:
            return result[0]["generated_text"]
        return ""
