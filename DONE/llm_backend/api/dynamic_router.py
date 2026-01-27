import logging
from fastapi import APIRouter, HTTPException
from .schemas import QueryRequest, QueryResponse
from .memory.in_memory import InMemoryRollingSummary
from ..core.applications.chat.graph_app import ChatGraphApp
from ..core.applications.rag.graph_app import RAGGraphApp
from .services.source_mapper import map_sources

logger = logging.getLogger(__name__)

def build_router(cfg):
    router = APIRouter()
    memory = InMemoryRollingSummary()

    try:
        if cfg.app_type == "chat":
            app = ChatGraphApp(cfg)
        elif cfg.app_type == "rag":
            app = RAGGraphApp(cfg)
        else:
            raise ValueError(f"Unknown app type: {cfg.app_type}")
    except Exception as e:
        logger.error(f"Failed to initialize app {cfg.app_name}: {e}")
        raise e


    @router.post("/query", response_model=QueryResponse)
    async def query(request: QueryRequest):
        try:
            summary = memory.get_summary(request.session_id)

            enriched_query = request.query
            if summary:
                enriched_query = f"""
Conversation summary:
{summary}

Current question:
{request.query}
"""

            # Run the app asynchronously
            result = await app.arun(enriched_query)

            memory.update(
                request.session_id,
                request.query,
                result["answer"],
                app.llm
            )

            return QueryResponse(
                answer=result["answer"],
                sources=map_sources(result.get("sources", []))
            )

        except Exception as e:
            logger.error(f"Error processing query for {cfg.app_name}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return router