# app/routes/users.py
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, Body, Path
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from app.core.dependencies import get_user_agent
from app.agents.user_agent import UserAgent
from app.services.database import get_user_service
from app.models.user import User, UserPreferences, UserPortfolio, UserAlert

router = APIRouter(prefix="/users", tags=["Users"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# Simple authentication for demonstration
async def get_current_user(token: str = Depends(oauth2_scheme)):
    user_service = get_user_service()
    user = await user_service.get_user_by_token(token)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


@router.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user_service = get_user_service()
    user = await user_service.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = await user_service.create_access_token(user.id)
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/register")
async def register_user(
    username: str = Body(...), email: str = Body(...), password: str = Body(...)
):
    """Register a new user."""
    user_service = get_user_service()

    # Check if user already exists
    existing_user = await user_service.get_user_by_username(username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")

    # Create new user
    user = await user_service.create_user(username, email, password)
    return {"message": "User registered successfully", "user_id": user.id}


@router.get("/me")
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information."""
    return current_user


@router.put("/me/preferences")
async def update_preferences(
    preferences: UserPreferences, current_user: User = Depends(get_current_user)
):
    """Update user preferences."""
    user_service = get_user_service()
    updated_user = await user_service.update_preferences(current_user.id, preferences)
    return {
        "message": "Preferences updated successfully",
        "preferences": updated_user.preferences,
    }


@router.get("/me/portfolio")
async def get_portfolio(current_user: User = Depends(get_current_user)):
    """Get user's investment portfolio."""
    user_service = get_user_service()
    portfolio = await user_service.get_portfolio(current_user.id)
    if not portfolio:
        return {"holdings": {}, "cash_balance": 0.0}
    return portfolio


@router.put("/me/portfolio")
async def update_portfolio(
    portfolio: UserPortfolio, current_user: User = Depends(get_current_user)
):
    """Update user's investment portfolio."""
    user_service = get_user_service()
    updated_portfolio = await user_service.update_portfolio(current_user.id, portfolio)
    return {"message": "Portfolio updated successfully", "portfolio": updated_portfolio}


@router.get("/me/alerts")
async def get_alerts(
    is_active: Optional[bool] = Query(None),
    current_user: User = Depends(get_current_user),
):
    """Get user's price alerts."""
    user_service = get_user_service()
    alerts = await user_service.get_alerts(current_user.id, is_active)
    return {"alerts": alerts}


@router.post("/me/alerts")
async def create_alert(
    alert: UserAlert, current_user: User = Depends(get_current_user)
):
    """Create a new price alert."""
    user_service = get_user_service()
    created_alert = await user_service.create_alert(current_user.id, alert)
    return {"message": "Alert created successfully", "alert": created_alert}


@router.delete("/me/alerts/{alert_id}")
async def delete_alert(
    alert_id: str = Path(...), current_user: User = Depends(get_current_user)
):
    """Delete a price alert."""
    user_service = get_user_service()
    success = await user_service.delete_alert(current_user.id, alert_id)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"message": "Alert deleted successfully"}


@router.post("/me/query")
async def process_user_query(
    query: str = Body(..., description="User's natural language query"),
    current_user: User = Depends(get_current_user),
    user_agent: UserAgent = Depends(get_user_agent),
):
    """Process a natural language query from the user."""
    try:
        response = await user_agent.process_query(query, user_id=current_user.id)

        # Save the interaction to the user's history
        user_service = get_user_service()
        await user_service.add_to_conversation_history(
            user_id=current_user.id,
            query=query,
            response=response.response,
            data=response.data,
        )

        return {
            "query": query,
            "response": response.response,
            "data": response.data,
            "timestamp": response.timestamp,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to process query: {str(e)}"
        )


@router.get("/me/history")
async def get_conversation_history(
    limit: int = Query(20, description="Number of recent interactions to return"),
    current_user: User = Depends(get_current_user),
):
    """Get user's conversation history."""
    user_service = get_user_service()
    history = await user_service.get_conversation_history(current_user.id, limit)
    return {"history": history}
