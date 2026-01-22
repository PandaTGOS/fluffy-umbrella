# Changed Files for Unified Enterprise Prompt Builder

Here are the complete contents of the files that were modified or created to implement the Unified Prompt Builder feature. Replacing the files with these contents will enable the feature.

## 1. `llm_backend/core/services/prompt_builder.py` (NEW)

```python
from typing import List, Optional
from llama_index.core.llms import ChatMessage, MessageRole

class PromptBuilder:
    """
    Enterprise-grade Prompt Builder to construct chat messages.
    Handles system prompts and user-defined prompt templates.
    """
    def __init__(self, system_prompt: str, prompt_template: Optional[str] = None):
        self.system_prompt = system_prompt
        # Default to passing the query directly if no template is provided
        self.prompt_template = prompt_template if prompt_template else "{query}"

    def build_messages(self, query: str, **kwargs) -> List[ChatMessage]:
        """
        Constructs a list of ChatMessages (System + User) based on the inputs.
        Accepts extra kwargs (like context_str) to format the template.
        """
        # 1. Format the user content using the template
        # Support both {query} and {query_str} for consistency with RAG
        # Also replace any extra kwargs provided
        user_content = self.prompt_template.replace("{query}", query).replace("{query_str}", query)
        
        for key, value in kwargs.items():
            user_content = user_content.replace(f"{{{key}}}", str(value))

        # 2. Construct messages
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=self.system_prompt),
            ChatMessage(role=MessageRole.USER, content=user_content)
        ]
        return messages
```

## 2. `llm_backend/core/app_config.py`

```python
from pydantic import BaseModel
from typing import Literal, Optional, Dict

class LLMConfig(BaseModel):
    provider: str
    model: str
    system_prompt: str
    # Unified prompt template for both Chat and RAG
    prompt_template: Optional[str] = None

class EmbeddingConfig(BaseModel):
    provider: str
    model: str

class VectorStoreConfig(BaseModel):
    type: Literal["in_memory", "pgvector"]
    collection: str

class RAGConfig(BaseModel):
    data_path: str
    chunk_size: int = 500
    top_k: int = 5
    # Deprecated: use llm.prompt_template instead
    qa_template: Optional[str] = None

class AppConfig(BaseModel):
    app_name: str
    app_type: Literal["chat", "rag"]

    llm: LLMConfig

    # 🔹 RAG-only (optional at schema level)
    embedding: Optional[EmbeddingConfig] = None
    vector_store: Optional[VectorStoreConfig] = None
    rag: Optional[RAGConfig] = None
    
    api: Dict
```

## 3. `llm_backend/core/adapters/openai_llm.py`

```python
import httpx
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

from ..interfaces import LLMProvider

class OpenAILLMProvider(LLMProvider):
    def __init__(self, model, system_prompt=None):
        self.model = model
        self.system_prompt = system_prompt

    def configure(self):
        # Allow system_prompt to be None if handled by PromptBuilder
        kwargs = {
            "model": self.model,
            "temperature": 0.1,
            "http_client": httpx.Client(verify=False)
        }
        if self.system_prompt:
            kwargs["system_prompt"] = self.system_prompt

        Settings.llm = OpenAI(**kwargs)
```

## 4. `llm_backend/core/applications/chat/graph_app.py`

```python
from langgraph.graph import StateGraph, END
from llama_index.core import Settings

from ...adapters.openai_llm import OpenAILLMProvider
from ...services.prompt_builder import PromptBuilder

def build_chat_app(llm, prompt_builder):
    graph = StateGraph(dict)

    def chat_node(state):
        # Use PromptBuilder to construct the full message history (System + User)
        messages = prompt_builder.build_messages(state["query"])
        response = llm.chat(messages)
        return {"answer": response.message.content}

    graph.add_node("chat", chat_node)
    graph.set_entry_point("chat")
    graph.add_edge("chat", END)

    return graph.compile()


class ChatGraphApp:
    def __init__(self, cfg):
        self.cfg = cfg

        # Initialize LLM without system_prompt attached to the driver
        # PromptBuilder handles the system prompt explicitly in the message history
        llm_provider = OpenAILLMProvider(
            model=cfg.llm.model,
            system_prompt=None 
        )
        llm_provider.configure()
        
        # Initialize PromptBuilder with system prompt definition and optional template
        # Use unified prompt_template from LLM config
        prompt_template = cfg.llm.prompt_template

        self.prompt_builder = PromptBuilder(
            system_prompt=cfg.llm.system_prompt,
            prompt_template=prompt_template
        )

        self.graph = build_chat_app(self.llm, self.prompt_builder)

    def run(self, query: str):
        return self.graph.invoke({"query": query})
```

## 5. `llm_backend/core/applications/rag/graph_app.py`

```python
from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from llama_index.core import Settings

from ...adapters.openai_llm import OpenAILLMProvider
from ...adapters.openai_embedding import OpenAIEmbeddingProvider
from ...adapters.memory_vector import InMemoryVectorStore
from ...services.document_loader import load_documents
from ...services.chunker import build_chunks

from ...services.prompt_builder import PromptBuilder

from .graph_nodes import (
    node_detect_language,
    node_retrieve,
    node_synthesize
)

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
        # Initialize LLM without system_prompt attached to the driver
        # PromptBuilder handles the system prompt explicitly in the message history
        llm_provider = OpenAILLMProvider(
            model=cfg.llm.model,
            system_prompt=None
        )
        llm_provider.configure()

        self.llm = Settings.llm
        
        # Initialize PromptBuilder (Unified)
        # Use simple fallback if prompt_template is missing, or rely on LLMConfig
        # Note: If no template is provided, we should probably have a default RAG template 
        # But PromptBuilder defaults to "{query}". We might want a smarter default for RAG if empty.
        template = cfg.llm.prompt_template or cfg.rag.qa_template # Fallback to qa_template for backward compat
        
        self.prompt_builder = PromptBuilder(
            system_prompt=cfg.llm.system_prompt,
            prompt_template=template
        )


        # Configure embeddings
        embed_provider = OpenAIEmbeddingProvider(cfg.embedding.model)
        embed_provider.configure()

        # Load documents
        documents = load_documents(cfg.rag.data_path)

        nodes, leaf_nodes = build_chunks(
            documents,
            chunk_sizes=[cfg.rag.chunk_size]
        )

        # Vector store
        store = InMemoryVectorStore()
        index, storage = store.build(nodes, leaf_nodes)

        # Build graph
        self.graph = build_rag_app(index, storage, self.cfg, self.prompt_builder, self.llm)


    def run(self, query: str):
        return self.graph.invoke({"query": query})


# =========================================================
# 3. GRAPH DEFINITION (PURE LANGGRAPH)
# =========================================================
def build_rag_app(index, storage, cfg, prompt_builder, llm):

    graph = StateGraph(RAGState)

    graph.add_node("detect_language", node_detect_language)
    graph.add_node(
        "retrieve_documents",
        lambda state: node_retrieve(state, index, storage, cfg)
    )
    graph.add_node(
        "synthesize",
        lambda state: node_synthesize(state, prompt_builder, llm)
    )

    graph.set_entry_point("detect_language")
    graph.add_edge("detect_language", "retrieve_documents")
    graph.add_edge("retrieve_documents", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()
```

## 6. `llm_backend/core/applications/rag/graph_nodes.py`

```python
from ...services.retriever import build_query_engine, detect_language

def is_in_scope(nodes):
    if not nodes:
        return False
    return len(" ".join(nodes).strip()) >= 60


def node_detect_language(state):
    return {"language": detect_language(state["query"])}


def node_retrieve(state, index, storage, cfg):
    engine = build_query_engine(
        state["query"],
        index,
        storage,
        top_k=cfg.rag.top_k,
        qa_template=cfg.rag.qa_template
    )

    retrieved = engine.retrieve(state["query"])

    contents, sources = [], set()
    for n in retrieved:
        contents.append(n.get_content())
        if "filename" in n.metadata:
            sources.add(n.metadata["filename"])

    return {
        "retrieved_nodes": contents,
        "sources": list(sources)
    }


def node_synthesize(state, prompt_builder, llm):
    """
    Synthesize answer using PromptBuilder and LLM chat interface.
    """
    if not is_in_scope(state["retrieved_nodes"]):
        return {"answer": ("This query is out of the HR scope, kindly contact the concerned HR person.")}

    # 1. Prepare context string
    context_str = "\n\n".join(state["retrieved_nodes"])

    # 2. Build messages using PromptBuilder (injecting context)
    messages = prompt_builder.build_messages(state["query"], context=context_str, context_str=context_str)

    # 3. Generate response
    response = llm.chat(messages)

    return {"answer": response.message.content}
```
