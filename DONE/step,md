# Enterprise Optimization Migration Guide (No Reranking)

Follow these steps to upgrade your original codebase to the optimized, async, decoupled architecture. This includes **Async Core**, **Decoupled Ingestion**, **Hybrid Search**, and **Security Middleware**.

---

## Step 1: Update Dependencies
Create or update `llm_backend/requirements.txt`:
```txt
fastapi>=0.109.0
uvicorn>=0.27.0
python-dotenv>=1.0.0
llama-index-core>=0.10.0
llama-index-llms-openai
llama-index-embeddings-openai
llama-index-vector-stores-postgres
asyncpg>=0.29.0
pgvector>=0.2.0
langgraph>=0.0.10
pydantic>=2.0.0
```

---

## Step 2: Update Configuration Support
Modify `llm_backend/core/app_config.py`.
**Changes**: Import `Field`, add `chunk_size` alias, add `use_hybrid_search` to `RAGConfig`.

```python
from pydantic import BaseModel, Field  # <--- Updated import
# ... imports ...

# ... LLMConfig, EmbeddingConfig ...

class RAGConfig(BaseModel):
    data_path: str
    chunk_sizes: List[int] = Field(default=[500], alias="chunk_size")  # <--- Added Alias
    top_k: int = 5
    use_hybrid_search: bool = False   # <--- Added Hybrid Search bool
    qa_template: str = ("...")

# ... AppConfig ...
```

---

## Step 3: Add `load_only` to PGVector Adapter
Modify `llm_backend/core/adapters/pg_vector.py`.
**Goal**: Allow connecting to the DB without trying to re-build/overwrite it.

```python
from typing import Iterable, Tuple, Optional # <--- Ensure Optional is imported

# ... inside PGVectorStore class ...

    # Add this new method at the end of the class
    def load_only(self) -> Tuple[VectorStoreIndex, StorageContext]:
        """
        Connect to existing Vector Store without modifying data.
        Used for Read-Only API Serving.
        """
        vector_store = self._build_vector_store()
        
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )

        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context
        )
        
        return index, storage_context
```

---

## Step 4: Enable Hybrid Search in Retriever
Modify `llm_backend/core/services/retriever.py`.
**Changes**: Accept `use_hybrid` param and configure retriever mode.

```python
def build_query_engine(
    query: str,
    index,
    storage,
    *,
    top_k: Optional[int] = None,
    qa_template: Optional[str] = None,
    use_hybrid: bool = False      # <--- New Param
):
    lang = detect_language(query)

    filters = MetadataFilters(filters=[
        MetadataFilter(key="language", value=lang)
    ])

    # Hybrid Search Logic
    vector_store_kwargs = {}
    if use_hybrid:
        vector_store_kwargs["vector_store_query_mode"] = "hybrid"

    base_retriever = index.as_retriever(
        similarity_top_k=top_k or DEFAULT_TOP_K,
        filters=filters,
        **vector_store_kwargs      # <--- Pass kwargs
    )

    # ... rest of function (AutoMergingRetriever, Synthesizer) ...
```

---

## Step 5: Convert Graph Nodes to Async
Modify `llm_backend/core/applications/rag/graph_nodes.py`.
**Changes**: `async def`, `await`, and pass `hybrid` config.

```python
# ... imports ...

# 1. Update build_query_engine call
async def node_synthesize(state, index, storage, cfg):  # <--- async
    # ... scope check ...

    engine = build_query_engine(
        state["query"],
        index,
        storage,
        top_k=cfg.rag.top_k,
        qa_template=cfg.rag.qa_template,
        use_hybrid=cfg.rag.use_hybrid_search  # <--- Pass config
    )

    answer = await engine.aquery(state["query"]) # <--- await aquery
    return {"answer": str(answer)}

async def node_retrieve(state, retriever):  # <--- async
    retrieved = await retriever.aretrieve(state["query"]) # <--- await aretrieve
    # ... processing ...

async def node_detect_language(state): # <--- async
    # ...

async def node_scope_guard(state): # <--- async
    # ... if not await guard.is_relevant(query): ...
```

---

## Step 6: Decouple & Async RAG App
Modify `llm_backend/core/applications/rag/graph_app.py`.
**Changes**: Skip ingestion if PGVector (Decoupling), Async run, Async nodes.

```python
class RAGGraphApp:
    def __init__(self, cfg):
        # ... setup ...

        # Vector Store Logic (Decoupling)
        if cfg.vector_store.type == "pgvector":
            store = PGVectorStore(...) # params
            try:
                index, storage = store.load_only() # <--- Try connecting first
            except Exception as e:
                raise RuntimeError(f"Run ingest script first! Error: {e}")
        elif cfg.vector_store.type == "in_memory":
            # Keep original build logic for dev/in-memory
            # ... load_documents(), build_chunks(), store.build() ...
        
        # ...

    async def arun(self, query: str):   # <--- Updated to arun
        return await self.graph.ainvoke({"query": query}) # <--- await ainvoke

# Update build_rag_app to use the async node functions above
# graph.add_node("retrieve_documents", retrieve_step_wrapper) 
# where retrieve_step_wrapper calls await node_retrieve(...)
```

---

## Step 7: Async Chat App
Modify `llm_backend/core/applications/chat/graph_app.py`.

```python
# ...
    async def chat_node(state):   # <--- async
        # ...
        response = await llm.achat(messages) # <--- await achat
        return {"answer": response.message.content}

# ...

class ChatGraphApp:
    # ...
    async def arun(self, query: str): # <--- async
        return await self.graph.ainvoke({"query": query})
```

---

## Step 8: Update Router & Middleware
1. Modify `llm_backend/api/dynamic_router.py`:
   - Use `await app.arun(...)`.
   - Add `try/except` and `logging`.

2. Modify `llm_backend/api/main.py`:
   - Add `CORSMiddleware`.
   - Add `RequestLoggingMiddleware` (see `ENTERPRISE_MIGRATION_LOG.md` for code).

---

## Step 9: Create Ingestion Script
Create `llm_backend/scripts/ingest.py`.
This file serves as the "Write" path. Copy the content exactly from the provided `ingest.py` artifact. It should:
1. Load `apps.yaml`.
2. check for `rag` type.
3. Call `load_documents`, `build_chunks`, and `store.build()`.

---

## Final Check
1. **Run Ingestion**: `python llm_backend/scripts/ingest.py`
2. **Run API**: `uvicorn llm_backend.api.main:app --reload`

Your code is now optimized without the Reranking complexity!
