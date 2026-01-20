from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage

from backend.core.registry import registry, AppConfig
from backend.domain.agents.builder import GraphBuilder
from backend.core.logger import get_logger

logger = get_logger(__name__)

class AgentService:
    def __init__(self):
        self._compiled_graphs: Dict[str, Any] = {}

    def get_agent(self, app_id: str):
        """
        Get or build the agent graph for a specific app_id.
        """
        if app_id not in self._compiled_graphs:
            logger.info(f"Building new agent for app_id: {app_id}")
            try:
                config = registry.get_app(app_id)
                builder = GraphBuilder(config)
                self._compiled_graphs[app_id] = builder.build()
            except Exception as e:
                logger.error(f"Failed to build agent for {app_id}: {e}")
                raise

        return self._compiled_graphs[app_id]

    async def chat(self, app_id: str, session_id: str, message: str, image_data: Optional[str] = None) -> str:
        """
        Chat with a specific app configuration, optionally with an image.
        """
        agent = self.get_agent(app_id)
        config = {"configurable": {"thread_id": session_id}}
        
        if image_data:
            content = [
                {"type": "text", "text": message},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]
        else:
            content = message

        input_messages = [HumanMessage(content=content)]
        
        response_content = ""
        # Using astream for async execution
        async for event in agent.astream({"messages": input_messages}, config, stream_mode="values"):
             if "messages" in event and event["messages"]:
                last_msg = event["messages"][-1]
                # We only want the AI response
                if last_msg.type == "ai":
                    response_content = last_msg.content
                
        return response_content
