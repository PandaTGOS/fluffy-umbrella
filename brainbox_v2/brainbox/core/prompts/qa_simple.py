from .builder import PromptSpec, PromptBuilder

class SimpleQAPrompt(PromptBuilder):
    def __init__(self, question: str):
        self.question = question

    def build(self) -> PromptSpec:
        return PromptSpec(
            system_instruction="Provide a helpful, concise, and honest answer.",
            user_input=self.question,
            context=None,
            output_schema=None,
        )
