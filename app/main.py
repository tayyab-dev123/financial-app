from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from app.database.mongodb import mongodb, get_database
from motor.motor_asyncio import AsyncIOMotorDatabase

app = FastAPI(title="Financial Assistant API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_db_client():
    await mongodb.connect()


@app.on_event("shutdown")
async def shutdown_db_client():
    await mongodb.close()


@app.get("/")
async def root():
    return {"message": "Financial Assistant API is running"}


@app.get("/test-db")
async def test_db(db: AsyncIOMotorDatabase = Depends(get_database)):
    # Test the database connection
    # Fix: Check if db is not None instead of using it in a boolean context
    db_status = {"connected": db is not None}

    # Additional check to prove connection works
    if db is not None:
        try:
            collections = await db.list_collection_names()
            db_status["collections"] = collections
        except Exception as e:
            db_status["error"] = str(e)

    return db_status
