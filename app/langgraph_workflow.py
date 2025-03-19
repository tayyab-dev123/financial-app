# app/langgraph_workflow.py
from typing import Dict, List, Any, TypedDict, Optional, Annotated
import json
from datetime import datetime

from langchain_core.runnables import RunnablePassthrough, Runnable, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langgraph.graph import StateGraph, END

from app.agents.data_agent import DataAgent
from app.agents.analysis_agent import AnalysisAgent
from app.agents.strategy_agent import StrategyAgent
from app.agents.user_agent import UserAgent
from app.models.stock import (
    StockData,
    StockAnalysis,
    SentimentAnalysis,
    TradingRecommendation,
)


# Define the state of our graph
class GraphState(TypedDict):
    """State for the financial assistant workflow."""

    # Input fields
    user_query: str
    user_id: Optional[str]
    symbols: List[str]
    query_type: str
    time_horizon: str
    risk_profile: str

    # Data from agents
    stock_data: Optional[Dict[str, StockData]]
    news_data: Optional[Dict[str, List[Dict[str, Any]]]]
    analysis_results: Optional[Dict[str, StockAnalysis]]
    sentiment_results: Optional[Dict[str, SentimentAnalysis]]
    recommendations: Optional[Dict[str, TradingRecommendation]]

    # Output field
    response: Optional[str]


def setup_graph() -> StateGraph:
    """Configure and return the LangGraph workflow for the financial assistant."""
    # Initialize agents
    data_agent = DataAgent()
    analysis_agent = AnalysisAgent()
    strategy_agent = StrategyAgent()
    user_agent = UserAgent()

    # Define the nodes (functions) of our graph

    # 1. Parse user query
    async def parse_query(state: GraphState) -> GraphState:
        """Parse the user's query to determine intent and extract symbols."""
        parsed_query = user_agent._parse_query(state["user_query"])

        return {
            **state,
            "symbols": parsed_query.symbols,
            "query_type": parsed_query.query_type,
            "time_horizon": parsed_query.time_horizon,
            "risk_profile": parsed_query.risk_profile,
        }

    # 2. Fetch stock data
    async def fetch_stock_data(state: GraphState) -> GraphState:
        """Fetch stock data for the identified symbols."""
        symbols = state["symbols"]
        time_horizon = state["time_horizon"]

        # Determine timeframe based on time horizon
        timeframe_map = {"short": "1H", "medium": "1D", "long": "1W"}
        timeframe = timeframe_map.get(time_horizon, "1D")

        # Fetch data for all symbols
        stock_data = {}
        for symbol in symbols:
            data = await data_agent.get_stock_data(symbol=symbol, timeframe=timeframe)
            stock_data[symbol] = data

        return {**state, "stock_data": stock_data}

    # 3. Fetch news data (if needed)
    async def fetch_news_data(state: GraphState) -> GraphState:
        """Fetch news data for the identified symbols if required by query type."""
        if state["query_type"] not in [
            "sentiment",
            "recommendation",
            "prediction",
            "general",
        ]:
            return state

        symbols = state["symbols"]
        news_data = {}

        for symbol in symbols:
            news = await data_agent.get_company_news(symbol)
            news_data[symbol] = news

        return {**state, "news_data": news_data}

    # 4. Run technical analysis
    async def analyze_stock_data(state: GraphState) -> GraphState:
        """Perform technical analysis on the stock data."""
        if not state.get("stock_data"):
            return state

        stock_data = state["stock_data"]
        analysis_results = {}

        for symbol, data in stock_data.items():
            analysis = await analysis_agent.analyze_stock_trend(data)
            analysis_results[symbol] = analysis

        return {**state, "analysis_results": analysis_results}

    # 5. Run sentiment analysis (if news data available)
    async def analyze_sentiment(state: GraphState) -> GraphState:
        """Perform sentiment analysis on news data."""
        if not state.get("news_data"):
            return state

        news_data = state["news_data"]
        sentiment_results = {}

        for symbol, news in news_data.items():
            sentiment = await analysis_agent.analyze_news_sentiment(news)
            sentiment_results[symbol] = sentiment

        return {**state, "sentiment_results": sentiment_results}

    # 6. Generate recommendations (if needed)
    async def generate_recommendations(state: GraphState) -> GraphState:
        """Generate trading recommendations based on analysis."""
        if state["query_type"] not in ["recommendation", "general"]:
            return state

        symbols = state["symbols"]
        analysis_results = state.get("analysis_results", {})
        sentiment_results = state.get("sentiment_results", {})
        risk_profile = state["risk_profile"]
        time_horizon = state["time_horizon"]

        recommendations = {}

        for symbol in symbols:
            if symbol in analysis_results:
                recommendation = await strategy_agent.generate_recommendation(
                    symbol=symbol,
                    stock_analysis=analysis_results[symbol],
                    sentiment_analysis=sentiment_results.get(symbol),
                    risk_profile=risk_profile,
                    investment_horizon=time_horizon,
                )
                recommendations[symbol] = recommendation

        return {**state, "recommendations": recommendations}

    # 7. Generate response
    async def generate_response(state: GraphState) -> GraphState:
        """Generate the final natural language response to the user."""
        # Prepare data for response generation
        response_data = {}
        context = {}

        # Add stock price data
        stock_data = state.get("stock_data", {})
        for symbol, data in stock_data.items():
            # Get latest price from the data
            if data and hasattr(data, "data") and not data.data.empty:
                latest_price = data.data["close"].iloc[-1]
                response_data[f"{symbol}_price"] = latest_price

        # Add analysis data
        analysis_results = state.get("analysis_results", {})
        for symbol, analysis in analysis_results.items():
            context[f"{symbol}_analysis"] = analysis
            response_data[f"{symbol}_trend"] = analysis.prediction
            response_data[f"{symbol}_indicators"] = analysis.indicators

        # Add sentiment data
        sentiment_results = state.get("sentiment_results", {})
        for symbol, sentiment in sentiment_results.items():
            context[f"{symbol}_sentiment"] = sentiment
            response_data[f"{symbol}_sentiment"] = {
                "score": sentiment.sentiment_score,
                "label": sentiment.overall_sentiment,
            }

        # Add recommendation data
        recommendations = state.get("recommendations", {})
        for symbol, recommendation in recommendations.items():
            context[f"{symbol}_recommendation"] = recommendation
            response_data[f"{symbol}_recommendation"] = {
                "action": recommendation.action,
                "entry_price": recommendation.entry_price,
                "stop_loss": recommendation.stop_loss,
                "take_profit": [target.price for target in recommendation.take_profit],
            }

        # Create a UserQuery object for response generation
        parsed_query = user_agent._parse_query(state["user_query"])

        # Generate the response
        response_text = await user_agent._generate_response(
            parsed_query, response_data, context
        )

        return {**state, "response": response_text}

    # Define the workflow graph
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("parse_query", parse_query)
    workflow.add_node("fetch_stock_data", fetch_stock_data)
    workflow.add_node("fetch_news_data", fetch_news_data)
    workflow.add_node("analyze_stock_data", analyze_stock_data)
    workflow.add_node("analyze_sentiment", analyze_sentiment)
    workflow.add_node("generate_recommendations", generate_recommendations)
    workflow.add_node("generate_response", generate_response)

    # Define edges (the flow of the workflow)
    workflow.add_edge("parse_query", "fetch_stock_data")
    workflow.add_edge("fetch_stock_data", "fetch_news_data")
    workflow.add_edge("fetch_news_data", "analyze_stock_data")
    workflow.add_edge("analyze_stock_data", "analyze_sentiment")
    workflow.add_edge("analyze_sentiment", "generate_recommendations")
    workflow.add_edge("generate_recommendations", "generate_response")
    workflow.add_edge("generate_response", END)

    # Set the entrypoint
    workflow.set_entry_point("parse_query")

    return workflow


# Create a compiled graph that can be used as a runnable
financial_assistant_graph = setup_graph().compile()


async def process_query(query: str, user_id: Optional[str] = None) -> str:
    """Process a user query through the financial assistant workflow."""
    # Initial state
    initial_state: GraphState = {
        "user_query": query,
        "user_id": user_id,
        "symbols": [],
        "query_type": "",
        "time_horizon": "",
        "risk_profile": "",
        "stock_data": None,
        "news_data": None,
        "analysis_results": None,
        "sentiment_results": None,
        "recommendations": None,
        "response": None,
    }

    # Run the workflow
    result = await financial_assistant_graph.ainvoke(initial_state)

    return result["response"]
