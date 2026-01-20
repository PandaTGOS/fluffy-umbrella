from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional
import base64
import logging

from backend.domain.agents.service import AgentService
from backend.core.config import settings

router = APIRouter()
agent_service = AgentService()
logger = logging.getLogger(__name__)

@router.post("/apps/{app_id}/ocr")
async def ocr_endpoint(
    app_id: str,
    file: UploadFile = File(...),
    session_id: str = Form(...),
    message: str = Form(default="Describe this image")
):
    """
    Multimodal endpoint to chat with an image (OCR).
    """
    try:
        # Read and encode image
        contents = await file.read()
        encoded_image = base64.b64encode(contents).decode("utf-8")
        
        response = await agent_service.chat(
            app_id=app_id,
            session_id=session_id,
            message=message,
            image_data=encoded_image
        )
        
        return {
            "response": response,
            "session_id": session_id,
            "app_id": app_id
        }
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(f"OCR Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
