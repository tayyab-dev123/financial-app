# app/agents/user_agent.py
from typing import Dict, List, Any, Optional, Tuple
import re
import json
from datetime import datetime, timedelta

from app.agents.data_agent import DataAgent
from app.agents.analysis_agent import AnalysisAgent
from app.agents.strategy_agent import StrategyAgent
from app.models.stock import UserQuery, AssistantResponse


class UserAgent:
    """Agent that interacts with user queries and orchestrates other agents."""

    def __init__(self):
        self.data_agent = DataAgent()
        self.analysis_agent = AnalysisAgent()
        self.strategy_agent = StrategyAgent()

    async def process_query(
        self, query: str, user_id: Optional[str] = None
    ) -> AssistantResponse:
        """
        Process a natural language query from the user.

        Args:
            query: User's natural language query
            user_id: Optional user identifier for personalization

        Returns:
            AssistantResponse with the appropriate data and explanation
        """
        # Parse the user query
        parsed_query = self._parse_query(query)

        # Fetch required data
        response_data, context = await self._fetch_data(parsed_query)

        # Generate natural language response
        response_text = await self._generate_response(
            parsed_query, response_data, context
        )

        return AssistantResponse(
            query=query,
            response=response_text,
            data=response_data,
            timestamp=datetime.now(),
        )

    def _parse_query(self, query: str) -> UserQuery:
        """Parse the user's natural language query to understand intent."""
        # Convert to lowercase for easier pattern matching
        query_lower = query.lower()

        # Extract stock symbols (assuming they're in uppercase in the original query)
        symbols = re.findall(r"\b[A-Z]{1,5}\b", query)

        # Default to analyzing SPY if no symbol found
        if not symbols:
            # Look for stock names
            common_stocks = {
                "apple": "AAPL",
                "microsoft": "MSFT",
                "amazon": "AMZN",
                "google": "GOOGL",
                "tesla": "TSLA",
                "facebook": "META",
                "meta": "META",
                "netflix": "NFLX",
                "spy": "SPY",
                "s&p": "SPY",
                "s&p 500": "SPY",
                "dow": "DIA",
                "nasdaq": "QQQ",
            }

            for name, symbol in common_stocks.items():
                if name in query_lower:
                    symbols = [symbol]
                    break

        # If still no symbol, use SPY as default
        if not symbols:
            symbols = ["SPY"]

        # Determine query type
        query_type = "general"

        if any(
            term in query_lower
            for term in ["buy", "sell", "invest", "recommendation", "should i"]
        ):
            query_type = "recommendation"
        elif any(
            term in query_lower for term in ["trend", "predict", "forecast", "outlook"]
        ):
            query_type = "prediction"
        elif any(term in query_lower for term in ["news", "sentiment"]):
            query_type = "sentiment"
        elif any(term in query_lower for term in ["chart", "technical", "indicator"]):
            query_type = "technical"
        elif any(term in query_lower for term in ["price", "value", "quote"]):
            query_type = "quote"

        # Determine time horizon
        time_horizon = "medium"  # Default

        if any(term in query_lower for term in ["day", "today", "short", "intraday"]):
            time_horizon = "short"
        elif any(term in query_lower for term in ["month", "year", "long"]):
            time_horizon = "long"

        # Determine risk profile (default to moderate)
        risk_profile = "moderate"

        if any(term in query_lower for term in ["safe", "conservative", "low risk"]):
            risk_profile = "conservative"
        elif any(term in query_lower for term in ["aggressive", "high risk", "risky"]):
            risk_profile = "aggressive"

        return UserQuery(
            raw_query=query,
            symbols=symbols,
            query_type=query_type,
            time_horizon=time_horizon,
            risk_profile=risk_profile,
        )

    async def _fetch_data(
        self, parsed_query: UserQuery
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Fetch required data based on the parsed query."""
        response_data = {}
        context = {}

        # Determine timeframe based on time horizon
        timeframe_map = {"short": "1H", "medium": "1D", "long": "1W"}
        timeframe = timeframe_map.get(parsed_query.time_horizon, "1D")

        # Determine date range based on time horizon
        end_date = datetime.now()

        if parsed_query.time_horizon == "short":
            start_date = end_date - timedelta(days=5)
        elif parsed_query.time_horizon == "medium":
            start_date = end_date - timedelta(days=90)
        else:  # long
            start_date = end_date - timedelta(days=365)

        # Fetch data for each symbol
        for symbol in parsed_query.symbols:
            # Get stock price data
            stock_data = await self.data_agent.get_stock_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
            )

            # Always get latest price
            latest_price = await self.data_agent.get_latest_price(symbol)
            response_data[f"{symbol}_price"] = latest_price

            # For all query types, perform basic analysis
            stock_analysis = await self.analysis_agent.analyze_stock_trend(stock_data)
            context[f"{symbol}_analysis"] = stock_analysis

            # Add selected analysis data to response
            response_data[f"{symbol}_trend"] = stock_analysis.prediction
            response_data[f"{symbol}_indicators"] = stock_analysis.indicators

            # If quote query, that's all we need
            if parsed_query.query_type == "quote":
                continue

            # Get news and sentiment for most query types
            if parsed_query.query_type in [
                "sentiment",
                "recommendation",
                "prediction",
                "general",
            ]:
                news_items = await self.data_agent.get_company_news(symbol)
                sentiment = await self.analysis_agent.analyze_news_sentiment(news_items)
                context[f"{symbol}_sentiment"] = sentiment
                response_data[f"{symbol}_sentiment"] = {
                    "score": sentiment.sentiment_score,
                    "label": sentiment.overall_sentiment,
                }

            # For recommendation queries, generate trading recommendation
            if parsed_query.query_type in ["recommendation", "general"]:
                recommendation = await self.strategy_agent.generate_recommendation(
                    symbol=symbol,
                    stock_analysis=stock_analysis,
                    sentiment_analysis=context.get(f"{symbol}_sentiment"),
                    risk_profile=parsed_query.risk_profile,
                    investment_horizon=parsed_query.time_horizon,
                )
                context[f"{symbol}_recommendation"] = recommendation
                response_data[f"{symbol}_recommendation"] = {
                    "action": recommendation.action,
                    "entry_price": recommendation.entry_price,
                    "stop_loss": recommendation.stop_loss,
                    "take_profit": [
                        target.price for target in recommendation.take_profit
                    ],
                }

        return response_data, context

    async def _generate_response(
        self,
        parsed_query: UserQuery,
        response_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> str:
        """Generate natural language response to the user's query."""
        # Start with a greeting
        response = "Here's what I found:\n\n"

        # Process each symbol
        for symbol in parsed_query.symbols:
            response += f"**{symbol}**: "

            # Basic price information
            if f"{symbol}_price" in response_data:
                response += (
                    f"Currently trading at ${response_data[f'{symbol}_price']:.2f}. "
                )

            # Add information based on query type
            if parsed_query.query_type == "quote":
                response += "Here are the latest price details."

            elif parsed_query.query_type == "technical":
                indicators = response_data.get(f"{symbol}_indicators", {})
                analysis = context.get(f"{symbol}_analysis")

                if analysis:
                    response += f"Technical analysis indicates a {analysis.prediction} outlook. "

                # Add key technical indicators
                if indicators:
                    rsi = indicators.get("rsi")
                    if rsi is not None:
                        response += f"RSI is at {rsi:.2f} "
                        if rsi > 70:
                            response += "(overbought). "
                        elif rsi < 30:
                            response += "(oversold). "
                        else:
                            response += "(neutral). "

                    macd = indicators.get("macd")
                    macd_signal = indicators.get("macd_signal")
                    if macd is not None and macd_signal is not None:
                        if macd > macd_signal:
                            response += "MACD is above signal line (bullish). "
                        else:
                            response += "MACD is below signal line (bearish). "

                    ma_50 = indicators.get("ma_50")
                    ma_200 = indicators.get("ma_200")
                    if ma_50 is not None and ma_200 is not None:
                        if ma_50 > ma_200:
                            response += "Price is above key moving averages (bullish). "
                        else:
                            response += "Price is below key moving averages (bearish). "

            elif parsed_query.query_type == "sentiment":
                sentiment_data = response_data.get(f"{symbol}_sentiment")
                if sentiment_data:
                    label = sentiment_data.get("label", "neutral")
                    score = sentiment_data.get("score", 0.5)

                    response += f"News sentiment is {label} "
                    response += f"with a score of {score:.2f}. "

                    sentiment_obj = context.get(f"{symbol}_sentiment")
                    if sentiment_obj and sentiment_obj.summary:
                        response += f"{sentiment_obj.summary} "

            elif parsed_query.query_type == "prediction":
                analysis = context.get(f"{symbol}_analysis")
                sentiment_data = response_data.get(f"{symbol}_sentiment")

                if analysis:
                    response += f"The {parsed_query.time_horizon}-term outlook is {analysis.prediction} "
                    response += f"with {analysis.confidence:.0%} confidence. "

                    if analysis.prediction == "bullish":
                        response += (
                            "Technical indicators suggest upward price movement. "
                        )
                    elif analysis.prediction == "bearish":
                        response += (
                            "Technical indicators suggest downward price movement. "
                        )
                    else:
                        response += (
                            "Technical indicators suggest sideways price movement. "
                        )

                if sentiment_data:
                    response += (
                        f"News sentiment is {sentiment_data.get('label', 'neutral')}, "
                    )
                    response += (
                        "which "
                        + (
                            "supports"
                            if sentiment_data.get("label") == analysis.prediction
                            else "contradicts"
                        )
                        + " the technical outlook. "
                    )

            elif parsed_query.query_type in ["recommendation", "general"]:
                recommendation = context.get(f"{symbol}_recommendation")

                if recommendation:
                    action_map = {
                        "strong_buy": "Strong Buy",
                        "buy": "Buy",
                        "hold": "Hold",
                        "sell": "Sell",
                        "strong_sell": "Strong Sell",
                    }

                    action = action_map.get(
                        recommendation.action, recommendation.action
                    )
                    response += f"**Recommendation: {action}**\n\n"

                    # Add reasoning
                    if recommendation.reasoning:
                        response += f"{recommendation.reasoning}\n\n"

                    # Add entry points and targets
                    if recommendation.action in [
                        "strong_buy",
                        "buy",
                        "strong_sell",
                        "sell",
                    ]:
                        response += "**Trading Strategy:**\n"

                        if recommendation.entry_price:
                            response += f"- Entry: ${recommendation.entry_price:.2f}\n"

                        if recommendation.stop_loss:
                            response += (
                                f"- Stop Loss: ${recommendation.stop_loss:.2f}\n"
                            )

                        if recommendation.take_profit:
                            response += "- Profit Targets:\n"
                            for idx, target in enumerate(
                                recommendation.take_profit[:3]
                            ):
                                response += f"  - Target {idx+1}: ${target.price:.2f}\n"

                        response += f"\nTimeframe: {recommendation.timeframe}\n\n"

            response += "\n"

        # Add disclaimer
        response += "\n*Disclaimer: This information is for educational purposes only and should not be considered financial advice. Always do your own research before making investment decisions.*"

        return response
