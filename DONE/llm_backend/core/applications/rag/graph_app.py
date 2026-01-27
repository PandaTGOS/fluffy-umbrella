from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from llama_index.core import Settings

from ...adapters.openai_llm import OpenAILLMProvider
from ...adapters.openai_embedding import OpenAIEmbeddingProvider
from ...adapters.memory_vector import InMemoryVectorStore
from ...adapters.pg_vector import PGVectorStore

from ...services.document_loader import load_documents
from ...services.chunker import build_chunks

from .graph_nodes import *

# =========================================================
# 1. GRAPH STATE (RAG-SPECIFIC)
# =========================================================
class RAGState(TypedDict):
    query: str
    language: Optional[str]
    retrieved_nodes: Optional[List[str]]
    answer: Optional[str]
    sources: Optional[List[str]]


# =========================================================
# 2. RAG GRAPH APPLICATION WRAPPER
# =========================================================
class RAGGraphApp:
    def __init__(self, cfg):
        self.cfg = cfg

        if not cfg.rag or not cfg.embedding or not cfg.vector_store:
            raise ValueError(
                "RAG app requires embedding, vector_store, and rag config"
            )


        # Configure LLM 
        llm_provider = OpenAILLMProvider(
            model=cfg.llm.model,
            system_prompt=cfg.llm.system_prompt
        )
        llm_provider.configure()

        self.llm = Settings.llm


        # Configure embeddings
        embed_provider = OpenAIEmbeddingProvider(cfg.embedding.model)
        embed_provider.configure()

        # Configure embeddings
        embed_provider = OpenAIEmbeddingProvider(cfg.embedding.model)
        embed_provider.configure()

        # Vector store configuration
        if cfg.vector_store.type == "pgvector":
            # Enterprise Mode: Connect to existing index (Read-Only)
            store = PGVectorStore(
                db_url=cfg.vector_store.db_url,
                table_name=cfg.vector_store.table_name,
                collection_name=cfg.vector_store.collection
            )
            try:
                index, storage = store.load_only()
            except Exception as e:
                 # Fallback for dev/testing if DB is empty? 
                 # Ideally enterprise should fail, but let's log warning.
                 raise RuntimeError(f"Failed to load existing index for {cfg.app_name}. Run ingestion script first! Error: {e}")

        elif cfg.vector_store.type == "in_memory":
            # Dev Mode: Build in-memory index
            store = InMemoryVectorStore()
            
            # Load documents (Expensive Startup!)
            documents = load_documents(cfg.rag.data_path)
            nodes, leaf_nodes = build_chunks(
                documents,
                chunk_sizes=cfg.rag.chunk_sizes
            )
            index, storage = store.build(nodes, leaf_nodes)

        else:
            raise ValueError(f"Unknown vector store type: {cfg.vector_store.type}")
        
        self.retriever = index.as_retriever(similarity_top_k=cfg.rag.top_k)

        # Build graph
        self.graph = build_rag_app(index, storage, self.cfg, self.retriever)


    async def arun(self, query: str):
        return await self.graph.ainvoke({"query": query})


# =========================================================
# 3. GRAPH DEFINITION (PURE LANGGRAPH)
# =========================================================
def build_rag_app(index, storage, cfg, retriever):

    graph = StateGraph(RAGState)

    async def retrieve_step(state):
        return await node_retrieve(state, retriever)

    async def synthesize_step(state):
        return await node_synthesize(state, index, storage, cfg)

    graph.add_node("detect_language", node_detect_language)
    graph.add_node("scope_guard", node_scope_guard)
    graph.add_node("retrieve_documents", retrieve_step)
    graph.add_node("synthesize", synthesize_step)


    #Graph Flow
    graph.set_entry_point("detect_language")
    graph.add_edge("detect_language", "scope_guard")

    graph.add_conditional_edges(
        "scope_guard",
        lambda state: "end" if state.get("blocked") else "retrieve",
        {
            "end": END,
            "retrieve": "retrieve_documents",
        }
    )

    graph.add_edge("retrieve_documents", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()