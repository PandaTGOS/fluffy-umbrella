from typing import Dict, Any, List
from .context import Context

def format_context_as_text(context: Context) -> str:
    """Formats the context documents into a single string."""
    if not context.documents:
        return ""
    
    formatted_docs = []
    for doc in context.documents:
        content = doc.get("content", "")
        source = doc.get("metadata", {}).get("source", "unknown")
        formatted_docs.append(f"Source: {source}\nContent: {content}")
        
    return "\n\n".join(formatted_docs)
