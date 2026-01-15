from ..interfaces import PromptSpec, PromptBuilder, Context

class RAGQAPrompt(PromptBuilder):
    def __init__(self, question: str, context: Context):
        self.question = question
        self.context = context

    def build(self) -> PromptSpec:
        system_instruction = (
            "You are a strict policy assistant. "
            "Your task is to answer the question using ONLY the provided context text. "
            "If the answer is not explicitly stated in the context, you MUST say 'I do not know' or 'The context does not contain this information'. "
            "DO NOT use any outside knowledge or general facts. "
            "DO NOT attempt to answer questions about geography, history, or general topics if not in the docs."
        )

        # Format context to be LLM-friendly (just text)
        formatted_context = []
        for doc in self.context.documents:
            # Handle both dict and object if needed, but context.documents is usually List[Dict]
            content = doc.get("content", "") if isinstance(doc, dict) else getattr(doc, "content", "")
            formatted_context.append(str(content))

        return PromptSpec(
            system_instruction=system_instruction,
            user_input=self.question,
            context=formatted_context,
            output_schema=None,
        )