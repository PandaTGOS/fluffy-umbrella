# RAG Fix & Parity Instructions

## Issue 1: Context Injection Failure (CRITICAL)
The RAG system was failing to inject context because `retriever.py` used incorrect template placeholders.

### Fix
Apply these changes to `llm_backend/core/services/retriever.py`:

**1. Update Default Template**
```python
# Change {query} -> {query_str} and {context} -> {context_str}
DEFAULT_QA_TEMPLATE = """...
Question: {query_str}

Context:
{context_str}
"""
```

**2. Fix Template Normalization Logic**
```python
def _normalize_template(template: Optional[str]) -> str:
    if not template:
        return DEFAULT_QA_TEMPLATE

    # Map custom placeholders TO standard ones
    template = template.replace("{context}", "{context_str}")
    template = template.replace("{query}", "{query_str}")
    template = template.replace("{question}", "{query_str}")

    if "{context_str}" not in template or "{query_str}" not in template:
        return DEFAULT_QA_TEMPLATE

    return template
```

---

## Issue 2: Configuration Mismatch (Parity with OLD)
The NEW system uses significantly different settings than the OLD system, resulting in worse performance:
- **Chunking**: NEW uses single-level `500` vs OLD's hierarchical `[2048, 512, 128]`.
- **Retrieval**: NEW uses `top_k=3` vs OLD's `top_k=10`.

To achieve parity (`NEW` behavior == `OLD` behavior), make the following changes:

### 1. Update Config Schema
Modify `llm_backend/core/app_config.py` to support list-based chunk sizes.

```python
from typing import List, Union # Add import

class RAGConfig(BaseModel):
    data_path: str
    # Change chunk_size: int to chunk_sizes: List[int]
    chunk_sizes: List[int] = [500] 
    top_k: int = 5
    qa_template: str = (...)
```

### 2. Update Graph Application
Modify `llm_backend/core/applications/rag/graph_app.py` to pass the list correctly.

```python
# Around line 60
nodes, leaf_nodes = build_chunks(
    documents,
    chunk_sizes=cfg.rag.chunk_sizes  # Remove the list brackets []
)
```

### 3. Update Configuration File
Modify `apps.yaml` to match the OLD system's values.

```yaml
    rag:
      data_path: ./data
      # chunk_size: 500  <-- DELETE
      chunk_sizes: [2048, 512, 128] # <-- ADD
      top_k: 10                     # <-- UPDATE from 3
      # ...
```

---

## Issue 3: Output Token Limit (Restricted Responses)
The NEW system has a hardcoded limit of **256 tokens**, which cuts off long answers.

## Issue 3: Output Token Limit (Restricted Responses)
The NEW system has a hardcoded limit of **256 tokens**, which cuts off long answers.

### Fix
Simply remove the `max_tokens` parameter to allow the model to determine the length (or use the provider's default maximum).

**Update Adapter (`llm_backend/core/adapters/openai_llm.py`)**

```python
    def configure(self):
        Settings.llm = OpenAI(
            model=self.model,
            system_prompt=self.system_prompt,
            # max_tokens=256,  <-- DELETE THIS LINE
            temperature=0.1,
            http_client=httpx.Client(verify=False)
        )
```

---

## Issue 4: Chat App Ignores System Prompt
You noticed that the `chat` app (e.g. brainrot persona) ignores the system prompt. This is because `llm.complete()` often treats input as simple user text without forcing the system prompt context.

### Fix
Update `llm_backend/core/applications/chat/graph_app.py` to explicitly construct the message history with screen prompt.

**1. Add Imports**
```python
from llama_index.core.llms import ChatMessage, MessageRole
```

**2. Update `chat_node`**
Replace the simple `llm.complete` call with a constructed `llm.chat` call.

```python
    def chat_node(state):
        # Force the system prompt to be included
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=llm.system_prompt),
            ChatMessage(role=MessageRole.USER, content=state["query"])
        ]
        response = llm.chat(messages)
        return {"answer": response.message.content}
```

---

## Feature: PGVector Support
To support swapping `in_memory` with `pgvector`, follow these steps:

### 1. Install Dependencies
```bash
pip install llama-index-vector-stores-postgres psycopg2-binary asyncpg
```

### 2. Update Config Schema (`llm_backend/core/app_config.py`)
Add database connection fields to `VectorStoreConfig`.

```python
class VectorStoreConfig(BaseModel):
    type: Literal["in_memory", "pgvector"]
    collection: str
    
    # Add PG params (optional, only used if type=pgvector)
    db_url: str = "postgresql://postgres:password@localhost:5432/vector_db"
    table_name: str = "embeddings"
```

### 3. Create PG Adapter (`llm_backend/core/adapters/pg_vector.py`)

```python
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore as LlamaPGVector
from ..interfaces import VectorStore

class PGVectorStore(VectorStore):
    def __init__(self, db_url, table_name, collection_name):
        self.db_url = db_url
        self.table_name = table_name
        self.collection_name = collection_name # Use as schema or part of table

    def build(self, nodes, leaf_nodes):
        # 1. Initialize PG Vector Store
        vector_store = LlamaPGVector.from_params(
            database=self.db_url.split("/")[-1],
            host=self.db_url.split("@")[1].split(":")[0],
            password=self.db_url.split(":")[2].split("@")[0],
            port=self.db_url.split(":")[-1].split("/")[0],
            user=self.db_url.split(":")[1].split("/")[2],
            table_name=self.table_name,
            embed_dim=1536  # Default for openai text-embedding-3-small
        )

        # 2. Create Storage Context
        storage = StorageContext.from_defaults(vector_store=vector_store)
        
        # 3. Add documents to docstore (metadata)
        storage.docstore.add_documents(nodes)

        # 4. Create Index (this pushes vectors to PG)
        index = VectorStoreIndex(leaf_nodes, storage_context=storage)
        
        return index, storage
```

### 4. Update RAG App (`llm_backend/core/applications/rag/graph_app.py`)
Update the vector store initialization section.

```python
# Import the new adapter
from ...adapters.pg_vector import PGVectorStore

# ... inside __init__ ...

        # Vector store configuration
        if cfg.vector_store.type == "in_memory":
            store = InMemoryVectorStore()
        elif cfg.vector_store.type == "pgvector":
            store = PGVectorStore(
                db_url=cfg.vector_store.db_url,
                table_name=cfg.vector_store.table_name,
                collection_name=cfg.vector_store.collection
            )
        else:
            raise ValueError(f"Unknown vector store type: {cfg.vector_store.type}")

        index, storage = store.build(nodes, leaf_nodes)
```

### 5. Config Usage (`apps.yaml`)
Now you can simply switch types.

```yaml
    vector_store:
      type: pgvector
      collection: hr_docs  # Can be used as logic grouping
      db_url: postgresql://user:pass@localhost:5432/mydb
      table_name: hr_embeddings
```

---

## note: System Prompt in RAG
For the **RAG** app, the "System Prompt" is effectively controlled by the `qa_template` in `apps.yaml`. The `llm.system_prompt` might be ignored by the LlamaIndex `CompactAndRefine` synthesizer in favor of the QA template.

**Ensure your `qa_template` includes your persona instructions.**

Example:
```yaml
    rag:
      qa_template: |
        You are a specialized legal assistant. Use ONLY the context below.
        
        Context:
        {context_str}
        
        Question:
        {query_str}
```


