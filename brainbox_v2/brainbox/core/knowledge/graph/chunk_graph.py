from typing import List
from brainbox.core.knowledge.documents import Document
from brainbox.core.knowledge.graph.graph_store import SimpleGraphStore

class ChunkGraph:
    """
    Builds a graph of document chunks.
    Links chunks by:
    1. Sequence (NEXT/PREV)
    2. Parent Document (PARENT)
    """
    def __init__(self):
        self.store = SimpleGraphStore()

    def build(self, documents: List[Document]):
        """
        Builds the graph from a list of documents.
        Assumes documents are chunks and might have metadata indicating sequence.
        """
        # Group by parent document if possible
        doc_groups = {}
        for doc in documents:
            parent_id = doc.metadata.get("parent_id", "root")
            if parent_id not in doc_groups:
                doc_groups[parent_id] = []
            doc_groups[parent_id].append(doc)

        for parent_id, docs in doc_groups.items():
            # Sort by sequence index if available
            docs.sort(key=lambda x: x.metadata.get("chunk_index", 0))
            
            for i, doc in enumerate(docs):
                self.store.add_node(doc.id, data=doc)
                
                # Link to parent
                self.store.add_edge(doc.id, parent_id, "CHILD_OF")
                self.store.add_edge(parent_id, doc.id, "HAS_CHILD")

                # Link sequential
                if i > 0:
                    prev_doc = docs[i-1]
                    self.store.add_edge(prev_doc.id, doc.id, "NEXT")
                    self.store.add_edge(doc.id, prev_doc.id, "PREV")

    def get_contextlist(self, doc_id: str, window: int = 1) -> List[Document]:
        """
        Get neighboring chunks (prev/next) for context window expansion.
        """
        context = []
        
        # Get PREV
        curr = doc_id
        for _ in range(window):
            neighbors = self.store.get_neighbors(curr, "PREV")
            if not neighbors:
                break
            prev_id, prev_data = neighbors[0] # Assume linear chain
            context.insert(0, prev_data)
            curr = prev_id

        # Add Self (if needed, but usually we have it)
        
        # Get NEXT
        curr = doc_id
        for _ in range(window):
            neighbors = self.store.get_neighbors(curr, "NEXT")
            if not neighbors:
                break
            next_id, next_data = neighbors[0]
            context.append(next_data)
            curr = next_id
            
        return context
