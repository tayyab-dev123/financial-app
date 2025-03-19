# app/agents/strategy_agent.py
from typing import Dict, List, Optional, Any
from datetime import datetime

from app.models.stock import StockAnalysis, SentimentAnalysis, TradingRecommendation
from app.services.trade_logic import TradeLogicService


class StrategyAgent:
    """Agent responsible for generating trading strategies and recommendations."""

    def __init__(self):
        self.trade_logic = TradeLogicService()

    async def generate_recommendation(
        self,
        symbol: str,
        stock_analysis: StockAnalysis,
        sentiment_analysis: Optional[SentimentAnalysis] = None,
        risk_profile: str = "moderate",
        investment_horizon: str = "medium",
    ) -> TradingRecommendation:
        """
        Generate a trading recommendation based on technical and sentiment analysis.

        Args:
            symbol: Stock ticker symbol
            stock_analysis: Technical analysis results
            sentiment_analysis: News sentiment analysis (optional)
            risk_profile: User's risk tolerance ('conservative', 'moderate', 'aggressive')
            investment_horizon: User's time horizon ('short', 'medium', 'long')

        Returns:
            TradingRecommendation object with action, reasoning, and targets
        """
        # Combine technical and sentiment signals
        signals = {}

        # Technical signals
        technical_score = 0

        # Trend prediction
        if stock_analysis.prediction == "bullish":
            technical_score += stock_analysis.confidence
            signals["trend_prediction"] = {
                "value": stock_analysis.prediction,
                "confidence": stock_analysis.confidence,
                "impact": "positive",
            }
        elif stock_analysis.prediction == "bearish":
            technical_score -= stock_analysis.confidence
            signals["trend_prediction"] = {
                "value": stock_analysis.prediction,
                "confidence": stock_analysis.confidence,
                "impact": "negative",
            }

        # RSI
        rsi = stock_analysis.indicators.get("rsi")
        if rsi is not None:
            if rsi > 70:  # Overbought
                technical_score -= 0.5
                signals["rsi"] = {
                    "value": rsi,
                    "interpretation": "overbought",
                    "impact": "negative",
                }
            elif rsi < 30:  # Oversold
                technical_score += 0.5
                signals["rsi"] = {
                    "value": rsi,
                    "interpretation": "oversold",
                    "impact": "positive",
                }

        # MACD
        macd = stock_analysis.indicators.get("macd")
        macd_signal = stock_analysis.indicators.get("macd_signal")
        if macd is not None and macd_signal is not None:
            if macd > macd_signal:  # Bullish
                technical_score += 0.3
                signals["macd"] = {"value": "bullish_crossover", "impact": "positive"}
            elif macd < macd_signal:  # Bearish
                technical_score -= 0.3
                signals["macd"] = {"value": "bearish_crossover", "impact": "negative"}

        # Moving averages
        ma_50 = stock_analysis.indicators.get("ma_50")
        ma_200 = stock_analysis.indicators.get("ma_200")
        if ma_50 is not None and ma_200 is not None:
            if ma_50 > ma_200:  # Golden cross or above
                technical_score += 0.4
                signals["moving_averages"] = {
                    "value": "price_above_mas",
                    "impact": "positive",
                }
            elif ma_50 < ma_200:  # Death cross or below
                technical_score -= 0.4
                signals["moving_averages"] = {
                    "value": "price_below_mas",
                    "impact": "negative",
                }

        # Normalize technical score to range [-1, 1]
        technical_score = max(min(technical_score, 1), -1)

        # Add sentiment score if available
        sentiment_score = 0
        if sentiment_analysis:
            # Map sentiment score from [0, 1] to [-1, 1]
            sentiment_score = (sentiment_analysis.sentiment_score - 0.5) * 2
            signals["news_sentiment"] = {
                "value": sentiment_analysis.overall_sentiment,
                "score": sentiment_analysis.sentiment_score,
                "confidence": sentiment_analysis.confidence,
                "impact": (
                    "positive"
                    if sentiment_analysis.sentiment_score > 0.5
                    else "negative"
                ),
            }

        # Combine technical and sentiment scores with adjustments for risk profile
        risk_multiplier = {
            "conservative": 0.5,  # More cautious
            "moderate": 1.0,  # Balanced
            "aggressive": 1.5,  # Higher risk tolerance
        }.get(risk_profile, 1.0)

        # Weight technical and sentiment scores
        # Technical analysis has higher weight for short-term, sentiment for long-term
        horizon_weights = {
            "short": (0.8, 0.2),  # 80% technical, 20% sentiment
            "medium": (0.6, 0.4),  # 60% technical, 40% sentiment
            "long": (0.4, 0.6),  # 40% technical, 60% sentiment
        }.get(investment_horizon, (0.6, 0.4))

        technical_weight, sentiment_weight = horizon_weights
        combined_score = (technical_score * technical_weight) + (
            sentiment_score * sentiment_weight
        )

        # Adjust based on risk profile
        adjusted_score = combined_score * risk_multiplier

        # Generate recommendation from the trade logic service
        recommendation = await self.trade_logic.generate_recommendation(
            symbol=symbol,
            adjusted_score=adjusted_score,
            current_price=stock_analysis.current_price,
            support_levels=stock_analysis.support_levels,
            resistance_levels=stock_analysis.resistance_levels,
            risk_profile=risk_profile,
            investment_horizon=investment_horizon,
            signals=signals,
        )

        return recommendation

    async def generate_portfolio_recommendations(
        self,
        portfolio: Dict[str, float],
        analyses: Dict[str, Dict[str, Any]],
        risk_profile: str = "moderate",
        cash_available: float = 0.0,
    ) -> List[TradingRecommendation]:
        """
        Generate recommendations for a portfolio of stocks.

        Args:
            portfolio: Dict mapping stock symbols to their weights in the portfolio
            analyses: Dict mapping symbols to their analysis results
            risk_profile: User's risk tolerance
            cash_available: Amount of cash available for new investments

        Returns:
            List of trading recommendations for portfolio management
        """
        recommendations = []

        # Generate individual recommendations
        for symbol, weight in portfolio.items():
            if symbol in analyses:
                stock_analysis = analyses[symbol].get("stock_analysis")
                sentiment_analysis = analyses[symbol].get("sentiment_analysis")

                if stock_analysis:
                    # Determine appropriate time horizon based on portfolio weight
                    # Higher weight = longer horizon, lower weight = shorter horizon
                    if weight > 0.2:  # Major position
                        horizon = "long"
                    elif weight > 0.05:  # Medium position
                        horizon = "medium"
                    else:  # Small position
                        horizon = "short"

                    recommendation = await self.generate_recommendation(
                        symbol=symbol,
                        stock_analysis=stock_analysis,
                        sentiment_analysis=sentiment_analysis,
                        risk_profile=risk_profile,
                        investment_horizon=horizon,
                    )

                    recommendations.append(recommendation)

        # If cash is available, find top opportunities
        if cash_available > 0:
            # Find stocks not in portfolio with highest potential
            opportunities = await self.find_investment_opportunities(
                analyses=analyses,
                excluded_symbols=list(portfolio.keys()),
                risk_profile=risk_profile,
                max_results=3,
            )

            recommendations.extend(opportunities)

        return recommendations

    async def find_investment_opportunities(
        self,
        analyses: Dict[str, Dict[str, Any]],
        excluded_symbols: List[str] = None,
        risk_profile: str = "moderate",
        max_results: int = 3,
    ) -> List[TradingRecommendation]:
        """
        Find new investment opportunities based on analysis results.

        Args:
            analyses: Dict mapping symbols to their analysis results
            excluded_symbols: List of symbols to exclude from results
            risk_profile: User's risk tolerance
            max_results: Maximum number of opportunities to return

        Returns:
            List of trading recommendations for potential investments
        """
        if excluded_symbols is None:
            excluded_symbols = []

        opportunities = []

        for symbol, analysis_data in analyses.items():
            if symbol in excluded_symbols:
                continue

            stock_analysis = analysis_data.get("stock_analysis")
            sentiment_analysis = analysis_data.get("sentiment_analysis")

            if (
                stock_analysis
                and stock_analysis.prediction == "bullish"
                and stock_analysis.confidence > 0.6
            ):
                recommendation = await self.generate_recommendation(
                    symbol=symbol,
                    stock_analysis=stock_analysis,
                    sentiment_analysis=sentiment_analysis,
                    risk_profile=risk_profile,
                    investment_horizon="medium",  # Default for new opportunities
                )

                # Only include strong buy recommendations for new opportunities
                if recommendation.action in ["strong_buy", "buy"]:
                    opportunities.append(recommendation)

        # Sort by confidence and limit results
        opportunities.sort(key=lambda x: x.confidence, reverse=True)
        return opportunities[:max_results]
