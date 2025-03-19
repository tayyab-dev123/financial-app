# app/services/trade_logic.py
from typing import Dict, List, Any, Optional
from datetime import datetime

from app.models.stock import TradingRecommendation, PriceTarget


class TradeLogicService:
    """Service that implements trading strategy logic and recommendation generation."""

    async def generate_recommendation(
        self,
        symbol: str,
        adjusted_score: float,
        current_price: float,
        support_levels: List[float],
        resistance_levels: List[float],
        risk_profile: str,
        investment_horizon: str,
        signals: Dict[str, Any],
    ) -> TradingRecommendation:
        """
        Generate a trading recommendation based on analysis results.

        Args:
            symbol: Stock ticker symbol
            adjusted_score: Combined and risk-adjusted score (-1.5 to 1.5)
            current_price: Current stock price
            support_levels: List of identified support price levels
            resistance_levels: List of identified resistance price levels
            risk_profile: User's risk profile ('conservative', 'moderate', 'aggressive')
            investment_horizon: User's investment horizon ('short', 'medium', 'long')
            signals: Dict containing all analysis signals and their interpretations

        Returns:
            TradingRecommendation object
        """
        # Determine action based on adjusted score
        action = self._determine_action(adjusted_score, risk_profile)

        # Calculate price targets
        entry_target = self._calculate_entry_price(
            current_price, action, support_levels, resistance_levels
        )

        # Calculate stop loss
        stop_loss = self._calculate_stop_loss(
            current_price, action, support_levels, resistance_levels, risk_profile
        )

        # Calculate take profit targets
        take_profit = self._calculate_take_profit(
            current_price,
            action,
            entry_target,
            resistance_levels,
            support_levels,
            risk_profile,
            investment_horizon,
        )

        # Generate reasoning
        reasoning = self._generate_reasoning(action, signals, adjusted_score)

        # Generate recommendation timeframe based on investment horizon
        timeframe_map = {"short": "days", "medium": "weeks", "long": "months"}
        recommendation_timeframe = timeframe_map.get(investment_horizon, "weeks")

        # Confidence level based on absolute value of adjusted score
        confidence = min(abs(adjusted_score), 1.0)

        return TradingRecommendation(
            symbol=symbol,
            action=action,
            current_price=current_price,
            entry_price=entry_target,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            reasoning=reasoning,
            timeframe=recommendation_timeframe,
            timestamp=datetime.now(),
        )

    def _determine_action(self, adjusted_score: float, risk_profile: str) -> str:
        """Determine the recommended action based on the adjusted score."""
        # Threshold adjustments based on risk profile
        thresholds = {
            "conservative": {
                "strong_buy": 0.8,
                "buy": 0.4,
                "sell": -0.4,
                "strong_sell": -0.8,
            },
            "moderate": {
                "strong_buy": 0.7,
                "buy": 0.3,
                "sell": -0.3,
                "strong_sell": -0.7,
            },
            "aggressive": {
                "strong_buy": 0.6,
                "buy": 0.2,
                "sell": -0.2,
                "strong_sell": -0.6,
            },
        }.get(
            risk_profile,
            {"strong_buy": 0.7, "buy": 0.3, "sell": -0.3, "strong_sell": -0.7},
        )

        if adjusted_score >= thresholds["strong_buy"]:
            return "strong_buy"
        elif adjusted_score >= thresholds["buy"]:
            return "buy"
        elif adjusted_score <= thresholds["strong_sell"]:
            return "strong_sell"
        elif adjusted_score <= thresholds["sell"]:
            return "sell"
        else:
            return "hold"

    def _calculate_entry_price(
        self,
        current_price: float,
        action: str,
        support_levels: List[float],
        resistance_levels: List[float],
    ) -> Optional[float]:
        """Calculate the recommended entry price."""
        if action in ["strong_buy", "buy"]:
            # For buys, suggest entry at current price or slight pullback
            if support_levels and support_levels[0] < current_price:
                # Entry between current price and nearest support
                return round((current_price + support_levels[0]) / 2, 2)
            else:
                # Small discount to current price
                return round(current_price * 0.99, 2)

        elif action in ["strong_sell", "sell"]:
            # For sells, suggest exit at current price or slight bounce
            if resistance_levels and resistance_levels[0] > current_price:
                # Exit between current price and nearest resistance
                return round((current_price + resistance_levels[0]) / 2, 2)
            else:
                # Small premium to current price
                return round(current_price * 1.01, 2)

        return None  # No entry for hold recommendation

    def _calculate_stop_loss(
        self,
        current_price: float,
        action: str,
        support_levels: List[float],
        resistance_levels: List[float],
        risk_profile: str,
    ) -> Optional[float]:
        """Calculate the recommended stop loss level."""
        # Risk factor based on risk profile
        risk_factor = {
            "conservative": 0.05,  # 5% loss tolerance
            "moderate": 0.08,  # 8% loss tolerance
            "aggressive": 0.12,  # 12% loss tolerance
        }.get(risk_profile, 0.08)

        if action in ["strong_buy", "buy"]:
            # For buys, stop loss below support or percentage-based
            if support_levels and support_levels[0] < current_price:
                # Set stop loss slightly below nearest support
                stop_loss = support_levels[0] * 0.99

                # Ensure stop loss is not too far from entry (risk management)
                max_distance = current_price * (1 - risk_factor)
                return round(max(stop_loss, max_distance), 2)
            else:
                # Percentage-based stop loss
                return round(current_price * (1 - risk_factor), 2)

        elif action in ["strong_sell", "sell"]:
            # For sells, stop loss above resistance or percentage-based
            if resistance_levels and resistance_levels[0] > current_price:
                # Set stop loss slightly above nearest resistance
                stop_loss = resistance_levels[0] * 1.01

                # Ensure stop loss is not too far from entry (risk management)
                max_distance = current_price * (1 + risk_factor)
                return round(min(stop_loss, max_distance), 2)
            else:
                # Percentage-based stop loss
                return round(current_price * (1 + risk_factor), 2)

        return None  # No stop loss for hold recommendation

    def _calculate_take_profit(
        self,
        current_price: float,
        action: str,
        entry_price: Optional[float],
        resistance_levels: List[float],
        support_levels: List[float],
        risk_profile: str,
        investment_horizon: str,
    ) -> List[PriceTarget]:
        """Calculate take profit targets with calculated rationales."""
        if action not in ["strong_buy", "buy", "strong_sell", "sell"]:
            return []  # No take profit for hold recommendation

        # Target multipliers based on investment horizon
        horizon_multipliers = {
            "short": [1.5, 2.0],  # Risk:reward ratios for short-term
            "medium": [2.0, 3.0, 4.0],  # Risk:reward ratios for medium-term
            "long": [3.0, 5.0, 8.0],  # Risk:reward ratios for long-term
        }.get(investment_horizon, [2.0, 3.0])

        # Additional multiplier based on risk profile
        risk_multiplier = {
            "conservative": 0.8,  # More conservative targets
            "moderate": 1.0,  # Standard targets
            "aggressive": 1.2,  # More aggressive targets
        }.get(risk_profile, 1.0)

        targets = []

        if action in ["strong_buy", "buy"]:
            # Entry price or current price if entry is None
            base_price = entry_price if entry_price is not None else current_price

            # Stop loss or default stop loss
            stop_loss = self._calculate_stop_loss(
                current_price, action, support_levels, resistance_levels, risk_profile
            )

            if stop_loss is None:
                stop_loss = base_price * 0.95  # Default 5% stop loss

            # Calculate risk per share
            risk_per_share = base_price - stop_loss

            # Generate targets based on risk:reward ratios
            for idx, multiplier in enumerate(horizon_multipliers):
                adjusted_multiplier = multiplier * risk_multiplier
                price_target = base_price + (risk_per_share * adjusted_multiplier)

                # Check if targets are near resistance levels and adjust
                if resistance_levels:
                    # Find nearest resistance level above price target
                    higher_resistances = [
                        r for r in resistance_levels if r > price_target
                    ]
                    if higher_resistances:
                        nearest_resistance = min(higher_resistances)
                        # If very close to resistance, set target slightly below it
                        if nearest_resistance / price_target < 1.05:  # Within 5%
                            price_target = (
                                nearest_resistance * 0.98
                            )  # Set just below resistance

                target = PriceTarget(
                    price=round(price_target, 2),
                    timeframe=f"{idx + 1}",
                    confidence=round(
                        0.9 / (idx + 1), 2
                    ),  # Reduced confidence for further targets
                )
                targets.append(target)

        elif action in ["strong_sell", "sell"]:
            # Similar logic for sell targets (in the opposite direction)
            base_price = entry_price if entry_price is not None else current_price

            stop_loss = self._calculate_stop_loss(
                current_price, action, support_levels, resistance_levels, risk_profile
            )

            if stop_loss is None:
                stop_loss = base_price * 1.05  # Default 5% stop loss

            risk_per_share = stop_loss - base_price

            for idx, multiplier in enumerate(horizon_multipliers):
                adjusted_multiplier = multiplier * risk_multiplier
                price_target = base_price - (risk_per_share * adjusted_multiplier)

                # Check if targets are near support levels and adjust
                if support_levels:
                    lower_supports = [s for s in support_levels if s < price_target]
                    if lower_supports:
                        nearest_support = max(lower_supports)
                        if price_target / nearest_support < 1.05:  # Within 5%
                            price_target = (
                                nearest_support * 1.02
                            )  # Set just above support

                target = PriceTarget(
                    price=round(price_target, 2),
                    timeframe=f"{idx + 1}",
                    confidence=round(0.9 / (idx + 1), 2),
                )
                targets.append(target)

        return targets

    def _generate_reasoning(
        self, action: str, signals: Dict[str, Any], adjusted_score: float
    ) -> str:
        """Generate reasoning for the recommendation."""
        if action == "hold":
            return "Mixed signals indicate a neutral outlook. Consider waiting for clearer direction."

        # Extract the most significant signals
        significant_signals = []

        # Process technical signals
        trend_signal = signals.get("trend_prediction", {})
        if trend_signal:
            if (
                trend_signal.get("impact") == "positive"
                and trend_signal.get("confidence", 0) > 0.6
            ):
                significant_signals.append(
                    f"Strong bullish trend detected with {trend_signal.get('confidence', 0):.0%} confidence"
                )
            elif (
                trend_signal.get("impact") == "negative"
                and trend_signal.get("confidence", 0) > 0.6
            ):
                significant_signals.append(
                    f"Strong bearish trend detected with {trend_signal.get('confidence', 0):.0%} confidence"
                )

        # Process RSI signal
        rsi_signal = signals.get("rsi", {})
        if rsi_signal:
            if rsi_signal.get("interpretation") == "overbought":
                significant_signals.append(
                    f"RSI indicates overbought conditions at {rsi_signal.get('value', 0):.1f}"
                )
            elif rsi_signal.get("interpretation") == "oversold":
                significant_signals.append(
                    f"RSI indicates oversold conditions at {rsi_signal.get('value', 0):.1f}"
                )

        # Process MACD signal
        macd_signal = signals.get("macd", {})
        if macd_signal:
            if macd_signal.get("value") == "bullish_crossover":
                significant_signals.append("MACD shows bullish crossover")
            elif macd_signal.get("value") == "bearish_crossover":
                significant_signals.append("MACD shows bearish crossover")

        # Process moving averages signal
        ma_signal = signals.get("moving_averages", {})
        if ma_signal:
            if ma_signal.get("value") == "price_above_mas":
                significant_signals.append("Price is trading above key moving averages")
            elif ma_signal.get("value") == "price_below_mas":
                significant_signals.append("Price is trading below key moving averages")

        # Process sentiment signal
        sentiment_signal = signals.get("news_sentiment", {})
        if sentiment_signal:
            if (
                sentiment_signal.get("value") == "positive"
                and sentiment_signal.get("score", 0) > 0.6
            ):
                significant_signals.append(
                    f"News sentiment is positive ({sentiment_signal.get('score', 0):.0%})"
                )
            elif (
                sentiment_signal.get("value") == "negative"
                and sentiment_signal.get("score", 0) < 0.4
            ):
                significant_signals.append(
                    f"News sentiment is negative ({sentiment_signal.get('score', 0):.0%})"
                )

        # Build reasoning based on action and signals
        action_phrases = {
            "strong_buy": "Strong buy recommendation based on",
            "buy": "Buy recommendation based on",
            "strong_sell": "Strong sell recommendation based on",
            "sell": "Sell recommendation based on",
        }

        opening = action_phrases.get(action, "Recommendation based on")

        # Add confidence level
        confidence = abs(adjusted_score)
        if confidence > 0.8:
            confidence_phrase = "high confidence"
        elif confidence > 0.5:
            confidence_phrase = "moderate confidence"
        else:
            confidence_phrase = "cautious confidence"

        # Combine signals into reasoning
        if significant_signals:
            signals_text = ": " + ", ".join(significant_signals[:3])
            if len(significant_signals) > 3:
                signals_text += ", and other factors"
            reasoning = f"{opening} {confidence_phrase}{signals_text}."
        else:
            reasoning = f"{opening} {confidence_phrase} in overall market analysis."

        return reasoning
