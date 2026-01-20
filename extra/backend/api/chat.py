from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

from backend.domain.agents.service import AgentService
from backend.core.config import settings
from backend.core.registry import registry
from backend.core.schemas import AppConfig

router = APIRouter()
agent_service = AgentService()

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    session_id: str
    app_id: str

@router.get("/apps", response_model=List[AppConfig])
async def list_apps():
    """List all available applications"""
    return registry.list_apps()

@router.post("/apps/{app_id}/chat", response_model=ChatResponse)
async def chat_with_app(app_id: str, request: ChatRequest):
    """
    Chat with a specific configured application.
    """
    try:
        response = await agent_service.chat(
            app_id=app_id,
            session_id=request.session_id,
            message=request.message
        )
        return ChatResponse(
            response=response, 
            session_id=request.session_id,
            app_id=app_id
        )
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
