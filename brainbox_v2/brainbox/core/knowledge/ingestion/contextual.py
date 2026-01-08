from typing import List
from brainbox.core.knowledge.documents import Document

class ContextualIngestion:
    """
    Adds document-level context to chunk content to ensure no chunk is meaningful out of context.
    """
    def transform(self, parent_doc: Document, chunks: List[Document]) -> List[Document]:
        """
        Augment each chunk with context from the parent document.
        """
        # A simple strategy: Prepend Title or Summary
        # "Title: X. Summary: Y. Chunk Content: ..."
        
        context_str = ""
        if "title" in parent_doc.metadata:
            context_str += f"Document Title: {parent_doc.metadata['title']}. "
        if "summary" in parent_doc.metadata:
            context_str += f"Context: {parent_doc.metadata['summary']}. "
            
        for chunk in chunks:
            # We modify content to include context
            # NOTE: We might want to keep original content separate from embedded content.
            # But for simplicity, we prepend to content.
            
            chunk.content = f"{context_str}\nContent: {chunk.content}"
            chunk.metadata["has_context"] = True
            
        return chunks
