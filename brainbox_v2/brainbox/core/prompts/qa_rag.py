from .builder import PromptSpec, PromptBuilder
from ..context import Context

class RAGQAPrompt(PromptBuilder):
    def __init__(self, question: str, context: Context):
        self.question = question
        self.context = context

    def build(self) -> PromptSpec:
        system_instruction = (
            "Answer using only the provided context. "
            "If the answer cannot be found in the context, say you do not know."
        )

        return PromptSpec(
            system_instruction=system_instruction,
            user_input=self.question,
            context=self.context.documents,
            output_schema=None,
        )
