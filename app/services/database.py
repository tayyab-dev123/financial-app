# app/services/database.py
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import os
import json
import uuid
import hashlib
import jwt
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import DESCENDING

from app.core.config import settings
from app.models.user import User, UserPreferences, UserPortfolio, UserAlert, UserSession


class DatabaseService:
    """Service for database operations."""

    def __init__(self):
        """Initialize the database connection."""
        self.client = AsyncIOMotorClient(settings.MONGODB_URL)
        self.db = self.client[settings.MONGODB_DATABASE]

        # Collections
        self.users = self.db.users
        self.portfolios = self.db.portfolios
        self.alerts = self.db.alerts
        self.sessions = self.db.sessions
        self.conversations = self.db.conversations

    # User management functions

    async def create_user(self, username: str, email: str, password: str) -> User:
        """Create a new user."""
        # Hash the password
        hashed_password = self._hash_password(password)

        # Create a new user record
        user_id = str(uuid.uuid4())
        user_data = {
            "id": user_id,
            "username": username,
            "email": email,
            "password_hash": hashed_password,
            "created_at": datetime.now(),
            "last_login": None,
            "preferences": UserPreferences().dict(),
        }

        # Insert into database
        await self.users.insert_one(user_data)

        # Return user object (without password)
        user_data.pop("password_hash")
        return User(**user_data)

    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user with username and password."""
        # Find the user
        user_data = await self.users.find_one({"username": username})
        if not user_data:
            return None

        # Check password
        if not self._verify_password(password, user_data["password_hash"]):
            return None

        # Update last login
        await self.users.update_one(
            {"id": user_data["id"]}, {"$set": {"last_login": datetime.now()}}
        )

        # Return user object (without password)
        user_data.pop("password_hash")
        return User(**user_data)

    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get a user by ID."""
        user_data = await self.users.find_one({"id": user_id})
        if not user_data:
            return None

        user_data.pop("password_hash", None)
        return User(**user_data)

    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get a user by username."""
        user_data = await self.users.find_one({"username": username})
        if not user_data:
            return None

        user_data.pop("password_hash", None)
        return User(**user_data)

    async def get_user_by_token(self, token: str) -> Optional[User]:
        """Get a user by authentication token."""
        try:
            payload = jwt.decode(
                token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
            )
            user_id = payload.get("sub")
            if user_id is None:
                return None
        except jwt.PyJWTError:
            return None

        return await self.get_user_by_id(user_id)

    async def create_access_token(self, user_id: str) -> str:
        """Create a JWT access token for a user."""
        expires_delta = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        expire = datetime.utcnow() + expires_delta

        to_encode = {"sub": user_id, "exp": expire}
        encoded_jwt = jwt.encode(
            to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM
        )
        return encoded_jwt

    async def update_preferences(
        self, user_id: str, preferences: UserPreferences
    ) -> User:
        """Update a user's preferences."""
        await self.users.update_one(
            {"id": user_id}, {"$set": {"preferences": preferences.dict()}}
        )

        return await self.get_user_by_id(user_id)

    # Portfolio management functions

    async def get_portfolio(self, user_id: str) -> Optional[UserPortfolio]:
        """Get a user's portfolio."""
        portfolio_data = await self.portfolios.find_one({"user_id": user_id})
        if not portfolio_data:
            return None

        return UserPortfolio(**portfolio_data)

    async def update_portfolio(
        self, user_id: str, portfolio: UserPortfolio
    ) -> UserPortfolio:
        """Update a user's portfolio."""
        portfolio_dict = portfolio.dict()
        portfolio_dict["last_updated"] = datetime.now()

        await self.portfolios.update_one(
            {"user_id": user_id}, {"$set": portfolio_dict}, upsert=True
        )

        return await self.get_portfolio(user_id)

    # Alert management functions

    async def get_alerts(
        self, user_id: str, is_active: Optional[bool] = None
    ) -> List[UserAlert]:
        """Get a user's alerts."""
        query = {"user_id": user_id}
        if is_active is not None:
            query["is_active"] = is_active

        cursor = self.alerts.find(query)
        alerts = []

        async for alert_data in cursor:
            alerts.append(UserAlert(**alert_data))

        return alerts

    async def create_alert(self, user_id: str, alert: UserAlert) -> UserAlert:
        """Create a new alert for a user."""
        alert_dict = alert.dict()
        alert_dict["id"] = str(uuid.uuid4())
        alert_dict["user_id"] = user_id
        alert_dict["created_at"] = datetime.now()

        await self.alerts.insert_one(alert_dict)

        return UserAlert(**alert_dict)

    async def delete_alert(self, user_id: str, alert_id: str) -> bool:
        """Delete a user's alert."""
        result = await self.alerts.delete_one({"id": alert_id, "user_id": user_id})
        return result.deleted_count > 0

    # Conversation history functions

    async def add_to_conversation_history(
        self, user_id: str, query: str, response: str, data: Dict[str, Any]
    ) -> None:
        """Add an interaction to the user's conversation history."""
        conversation_entry = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "query": query,
            "response": response,
            "data": data,
            "timestamp": datetime.now(),
        }

        await self.conversations.insert_one(conversation_entry)

    async def get_conversation_history(
        self, user_id: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get a user's conversation history."""
        cursor = (
            self.conversations.find({"user_id": user_id})
            .sort("timestamp", DESCENDING)
            .limit(limit)
        )

        history = []
        async for entry in cursor:
            entry["_id"] = str(entry["_id"])  # Convert ObjectId to string
            history.append(entry)

        # Reverse to get chronological order
        return list(reversed(history))

    # Helper methods

    def _hash_password(self, password: str) -> str:
        """Hash a password using SHA-256."""
        return hashlib.sha256((password + settings.PASSWORD_SALT).encode()).hexdigest()

    def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return self._hash_password(plain_password) == hashed_password


# Global instance
_db_service = None


def get_user_service() -> DatabaseService:
    """Get the database service singleton."""
    global _db_service
    if _db_service is None:
        _db_service = DatabaseService()
    return _db_service
