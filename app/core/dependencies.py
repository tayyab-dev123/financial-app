# app/core/dependencies.py
from typing import Optional
from fastapi import Depends

from app.agents.data_agent import DataAgent
from app.agents.analysis_agent import AnalysisAgent
from app.agents.strategy_agent import StrategyAgent
from app.agents.user_agent import UserAgent
from app.services.stock_data_factory import StockDataFactory


# Singleton instances
_data_agent = None
_analysis_agent = None
_strategy_agent = None
_user_agent = None


def get_data_agent() -> DataAgent:
    """Get the data agent singleton."""
    global _data_agent
    if _data_agent is None:
        stock_service = StockDataFactory.create_service()
        _data_agent = DataAgent(stock_service=stock_service)
    return _data_agent


def get_analysis_agent() -> AnalysisAgent:
    """Get the analysis agent singleton."""
    global _analysis_agent
    if _analysis_agent is None:
        _analysis_agent = AnalysisAgent()
    return _analysis_agent


def get_strategy_agent() -> StrategyAgent:
    """Get the strategy agent singleton."""
    global _strategy_agent
    if _strategy_agent is None:
        _strategy_agent = StrategyAgent()
    return _strategy_agent


def get_user_agent() -> UserAgent:
    """Get the user agent singleton."""
    global _user_agent
    if _user_agent is None:
        _user_agent = UserAgent()
    return _user_agent
