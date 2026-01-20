from fastapi import FastAPI
from backend.core.config import settings
from backend.core.logger import get_logger

logger = get_logger(__name__)

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        description="Unified Enterprise LLM Backend Platform"
    )

    @app.get("/health")
    async def health_check():
        return {
            "status": "active",
            "environment": settings.ENVIRONMENT,
            "version": "0.1.0"
        }

    from backend.api.chat import router as chat_router
    from backend.api.rag import router as rag_router
    from backend.api.ocr import router as ocr_router
    
    app.include_router(chat_router, prefix=f"{settings.API_V1_STR}", tags=["chat"])
    app.include_router(rag_router, prefix=f"{settings.API_V1_STR}/rag", tags=["rag"])
    app.include_router(ocr_router, prefix=f"{settings.API_V1_STR}", tags=["ocr"])

    logger.info(f"Application initialized in {settings.ENVIRONMENT} mode")
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
