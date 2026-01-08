from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from brainbox.core.knowledge.documents import Document
from brainbox.core.knowledge.retrieval_result import RetrievalResult
from brainbox.core.knowledge.retrievers.base import BaseRetriever
from brainbox.core.vectorstore.chroma import ChromaVectorStore
from brainbox.core.knowledge.indexing.inverted_index import InvertedIndex
from brainbox.core.knowledge.retrievers.inverted_index import InvertedIndexRetriever
from brainbox.core.security.rbac import RBACManager, User
from brainbox.core.routing.prefix_router import PrefixRouter
from brainbox.core.knowledge.graph.chunk_graph import ChunkGraph

@dataclass
class SecurePipelineResult:
    response: str
    sources: List[Document]
    signals: Dict[str, Any]

class SecureEnterpriseRAG:
    """
    A unified pipeline implementing Secure Enterprise LLM.
    Integrates:
    - Routing / Abuse Prevention
    - RBAC
    - Hybrid Retrieval (Vector + Inverted)
    - Graph-based Context Expansion
    """
    def __init__(self, 
                 vector_store: ChromaVectorStore,
                 inverted_index: InvertedIndex,
                 rbac: RBACManager,
                 router: PrefixRouter,
                 graph: ChunkGraph):
        self.vector_store = vector_store
        self.inverted_retriever = InvertedIndexRetriever(inverted_index)
        self.rbac = rbac
        self.router = router
        self.graph = graph

    def run(self, query: str, user: User) -> SecurePipelineResult:
        signals = {}

        # 1. Routing & Security Checks
        if self.router.check_abuse(query):
             return SecurePipelineResult("Query blocked due to safety policy.", [], {"blocked": True})
        
        route = self.router.route(query)
        signals["route"] = route
        
        if not self.rbac.can(user, "read"):
             return SecurePipelineResult("Access Denied.", [], {"access_denied": True})

        # 2. Hybrid Retrieval
        # Vector Search
        vector_res = self.vector_store.retrieve(query, k=5)
        # Inverted Index Search
        keyword_res = self.inverted_retriever.retrieve(query, k=5)
        
        # Combine Results (Naive Merge)
        combined_docs = {d.id: d for d in vector_res.documents + keyword_res.documents}.values()
        combined_docs = list(combined_docs)
        
        # 3. RBAC Filtering on Documents
        allowed_docs = self.rbac.filter_documents(user, combined_docs)
        signals["docs_filtered_out"] = len(combined_docs) - len(allowed_docs)
        
        if not allowed_docs:
             return SecurePipelineResult("No accessible documents found.", [], signals)

        # 4. Graph Expansion (Context/Chunking)
        # Expand context for top ranked doc
        top_doc = allowed_docs[0]
        context_chunks = self.graph.get_contextlist(top_doc.id, window=1)
        # Merge context chunks into source list if not present
        final_docs = [top_doc] + [c for c in context_chunks if c.id != top_doc.id]
        
        signals["context_expanded"] = len(context_chunks)

        # 5. Generation (Mock)
        # usage of LLM would go here
        context_text = "\n---\n".join([d.content for d in final_docs])
        response = f"Simulated Answer based on context:\n{context_text[:200]}..."
        
        return SecurePipelineResult(
            response=response, 
            sources=final_docs, 
            signals=signals
        )
