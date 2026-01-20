from typing import Any, Dict
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage

from backend.core.schemas import AppConfig, RAGConfig
from backend.infrastructure.llm.factory import LLMFactory
from backend.domain.rag.service import RAGService
from backend.core.logger import get_logger

logger = get_logger(__name__)

class GraphBuilder:
    def __init__(self, config: AppConfig):
        self.config = config
        self.llm = LLMFactory.get_chat_model(
            provider=config.llm.provider,
            model_name=config.llm.model,
            temperature=config.llm.temperature
        )
        if config.rag.enabled:
            from backend.infrastructure.vector.postgres import PostgresVectorStore
            from langchain_ollama import OllamaEmbeddings
            from backend.core.config import settings

            store = PostgresVectorStore()
            embeddings = OllamaEmbeddings(
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.DEFAULT_EMBEDDING_MODEL
            )
            self.rag_service = RAGService(vector_store=store, embeddings=embeddings)
        else:
            self.rag_service = None

    async def _node_retrieve(self, state: MessagesState):
        """
        Retrieval node: looks at the last user message and retrieves context.
        Injects context as a SystemMessage or modifies the last HumanMessage.
        For simplicity, we prepend a SystemMessage with context.
        """
        last_message = state["messages"][-1]
        query = last_message.content
        
        context = await self.rag_service.retrieve(query, k=self.config.rag.k)
        
        if context:
            # We add a hidden system message with context
            # Or formatted prompt. Let's send it as a separate message for now
            return {"messages": [SystemMessage(content=f"Context:\n{context}")]}
        return {}

    async def _node_generate(self, state: MessagesState):
        # Ensure system prompt is present
        # Ideally this is done at the start of conversation or we rely on the prompt being in history
        # But we can also prepend it here if not present.
        # LangChain models usually handle this, but let's be explicit
        
        messages = state["messages"]
        # Check if first message is system, if not, prepend config system prompt
        if not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=self.config.system_prompt)] + messages
            
        response = await self.llm.ainvoke(messages)
        return {"messages": response}

    def build(self):
        workflow = StateGraph(MessagesState)
        
        workflow.add_node("generate", self._node_generate)
        
        entry_point = "generate"
        
        if self.config.rag.enabled:
            workflow.add_node("retrieve", self._node_retrieve)
            workflow.add_edge("retrieve", "generate")
            entry_point = "retrieve"
        else:
            workflow.add_edge("generate", END)

        workflow.add_edge(START, entry_point)
        
        # Checkpointer
        memory = MemorySaver()
        
        logger.info(f"Built graph for app {self.config.id} (RAG: {self.config.rag.enabled})")
        return workflow.compile(checkpointer=memory)
