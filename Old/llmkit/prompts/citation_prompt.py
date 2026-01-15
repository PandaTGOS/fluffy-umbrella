from typing import List, Dict, Any
from ..interfaces import PromptSpec, PromptBuilder, Context

class CitationQAPrompt(PromptBuilder):
    def __init__(self, question: str, context: Context):
        self.question = question
        self.context = context

    def build(self) -> PromptSpec:
        system_instruction = (
            "You are a helpful and precise assistant. "
            "Answer the question using ONLY the provided context. "
            "You MUST cite the source of your information using the format [doc_id]. "
            "For example: 'The leave policy states 20 days [doc_1].' "
            "If the answer is not in the context, state that you do not know."
        )

        # Format context specifically for citation
        formatted_context = []
        doc_map = {}
        for i, doc in enumerate(self.context.documents):
            # Use a simpler ID for citation if doc['id'] is long
            # Or just use doc['id']. Let's use simplified numbered IDs for readability if needed,
            # but usually doc ID is good. 
            # Let's map internal ID to citation ID [doc_N] for consistency in prompt.
            citation_id = f"doc_{i+1}"
            content = doc.get('content', '')
            formatted_context.append(f"[{citation_id}] {content}")
            
            # (Optional) We could store the mapping if we wanted to reverse it later.
        
        return PromptSpec(
            system_instruction=system_instruction,
            user_input=self.question,
            context=formatted_context, 
            output_schema=None,
        )
