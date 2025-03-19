# app/database/mongodb.py
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings  # Changed from config to settings


class MongoDB:
    def __init__(self):
        self.client = None
        self.db = None

    async def connect(self):
        """Initialize MongoDB Connection"""
        self.client = AsyncIOMotorClient(settings.MONGO_URI)
        self.db = self.client[settings.MONGO_DB_NAME]  # Use the configured DB name
        print(f"Connected to MongoDB at {settings.MONGO_URI}")

    async def close(self):
        """Close MongoDB Connection"""
        if self.client:
            self.client.close()
            print("MongoDB connection closed")

    def get_db(self):
        """Get database instance"""
        return self.db


mongodb = MongoDB()


# FastAPI dependency to access the database
async def get_database():
    """Dependency for FastAPI to get database instance"""
    return mongodb.db
