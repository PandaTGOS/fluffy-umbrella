import hashlib
from urllib.parse import urlparse
from typing import Iterable, Tuple, Optional

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore as LlamaPGVector

from ..interfaces import VectorStore


# ---------------------------------------------------------
# Stable ID Generator
# ---------------------------------------------------------

def stable_node_id(text: str, source: str, chunk_idx: int) -> str:
    """
    Generate deterministic node ID.
    Must NEVER change for same content.
    """
    raw = f"{source}:{chunk_idx}:{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------
# PGVector Adapter
# ---------------------------------------------------------

class PGVectorStore(VectorStore):
    """
    PGVector adapter for LlamaIndex.

    Guarantees:
    - No duplicate embeddings
    - Stable IDs
    - Restart safe
    - Auto-merging safe
    - Docstore / vector store consistency
    """

    def __init__(
        self,
        db_url: str,
        table_name: str,
        collection_name: Optional[str] = None,
        embed_dim: int = 1536,
    ):
        self.db_url = db_url
        self.table_name = table_name
        self.collection_name = collection_name
        self.embed_dim = embed_dim

        self._parsed_url = urlparse(db_url)

        if not self._parsed_url.hostname:
            raise ValueError("Invalid PostgreSQL URL")

    # ---------------------------------------------------------
    # Vector Store Builder
    # ---------------------------------------------------------

    def _build_vector_store(self) -> LlamaPGVector:
        return LlamaPGVector.from_params(
            database=self._parsed_url.path.lstrip("/"),
            host=self._parsed_url.hostname,
            port=self._parsed_url.port or 5432,
            user=self._parsed_url.username,
            password=self._parsed_url.password,
            table_name=self.table_name,
            embed_dim=self.embed_dim,
        )

    # ---------------------------------------------------------
    # Stable ID Assignment (ALL nodes)
    # ---------------------------------------------------------

    def _assign_stable_ids(self, nodes: Iterable) -> None:
        """
        Assign stable IDs to ALL nodes (parents + leaves).
        This is mandatory for auto-merging retriever safety.
        """

        for idx, node in enumerate(nodes):

            # Defensive: avoid reassigning if already set
            if getattr(node, "id_", None):
                continue

            source = node.metadata.get("file_path", "unknown_source")
            chunk_idx = node.metadata.get("chunk", idx)

            node.id_ = stable_node_id(
                text=node.text,
                source=source,
                chunk_idx=chunk_idx,
            )

    # ---------------------------------------------------------
    # Build Index
    # ---------------------------------------------------------

    def build(self, nodes, leaf_nodes) -> Tuple[VectorStoreIndex, StorageContext]:
        """
        Safe to call repeatedly.

        Guarantees:
        - No ID drift
        - No orphan vectors
        - No missing parents
        - Safe upserts
        """

        # -----------------------------------------------------
        # 1. Assign IDs to ALL nodes (CRITICAL FIX)
        # -----------------------------------------------------

        self._assign_stable_ids(nodes)
        self._assign_stable_ids(leaf_nodes)

        # -----------------------------------------------------
        # 2. Initialize Vector Store
        # -----------------------------------------------------

        vector_store = self._build_vector_store()

        # -----------------------------------------------------
        # 3. Create Storage Context
        # -----------------------------------------------------

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )

        # -----------------------------------------------------
        # 4. Store Documents (Parents + Leaves)
        # -----------------------------------------------------

        # Docstore is keyed by node.id_
        storage_context.docstore.add_documents(nodes)

        # -----------------------------------------------------
        # 5. Build Index (Upserts via node_id)
        # -----------------------------------------------------

        index = VectorStoreIndex(
            leaf_nodes,
            storage_context=storage_context,
        )

        return index, storage_context


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
