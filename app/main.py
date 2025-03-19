from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.config import settings
from app.database.mongodb import mongodb, get_database
from app.routes import stocks, analysis, strategy, users
from app.langgraph_workflow import process_query

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="An AI-powered financial assistant for stock market analysis and recommendations",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Database startup and shutdown events
@app.on_event("startup")
async def startup_db_client():
    await mongodb.connect()


@app.on_event("shutdown")
async def shutdown_db_client():
    await mongodb.close()


# Include routers
app.include_router(stocks.router)
app.include_router(analysis.router)
app.include_router(strategy.router)
app.include_router(users.router)


@app.get("/")
async def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "message": "Welcome to the AI Financial Assistant API",
    }


@app.get("/test-db")
async def test_db(db: AsyncIOMotorDatabase = Depends(get_database)):
    # Test the database connection
    db_status = {"connected": db is not None}

    # Additional check to prove connection works
    if db is not None:
        try:
            collections = await db.list_collection_names()
            db_status["collections"] = collections
        except Exception as e:
            db_status["error"] = str(e)

    return db_status


@app.post("/query")
async def query(text: str):
    """Process a natural language query - no authentication required."""
    response = await process_query(text)
    return {"query": text, "response": response}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=9000, reload=True)
