from typing import Dict, Any, Type
from llmkit.interfaces import PromptSpec, PromptBuilder, LLMClient, Pipeline
from llmkit.prompts import SimpleQAPrompt

class SimplePipeline(Pipeline):
    def __init__(
        self, 
        client: LLMClient, 
        prompt_builder: Type[PromptBuilder] = SimpleQAPrompt
    ):
        self.client = client
        self.prompt_builder = prompt_builder

    def run(self, input_data: str) -> Dict[str, Any]:
        question = input_data
        
        # Build Prompt
        prompt_builder = self.prompt_builder(question=question)
        prompt_spec = prompt_builder.build()

        # Execute
        response = self.client.generate(
            system_instruction=prompt_spec.system_instruction,
            user_input=prompt_spec.user_input,
            context=prompt_spec.context,
            output_schema=prompt_spec.output_schema,
        )

        return {
            "question": question,
            "answer": response.text,
            "model_name": response.model_name,
            "token_usage": response.token_usage,
        }