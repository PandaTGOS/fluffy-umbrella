from typing import Optional
from ollama import Client
from .llm_client import LLMClient, LLMResponse

class OllamaClient(LLMClient):
    def __init__(self, host: Optional[str] = None, default_model: str = "deepscaler"):
        self._client = Client(host=host)
        self.default_model = default_model

    def generate(
        self,
        system_instruction: str,
        user_input: str,
        context: Optional[dict] = None,
        output_schema: Optional[object] = None,
        runtime_options: Optional[dict] = None,
    ) -> LLMResponse:
        
        runtime_options = runtime_options or {}

        model = runtime_options.get("model", self.default_model)
        temperature = runtime_options.get("temperature", 0.0)
        max_tokens = runtime_options.get("max_tokens", None)

        # Combine context and user input safely
        if context:
            if isinstance(context, dict):
                context_text = "\n".join(f"{k}: {v}" for k, v in context.items())
            elif isinstance(context, list):
                context_text = "\n".join(str(item) for item in context)
            else:
                context_text = str(context)
            user_content = f"{context_text}\n\n{user_input}"
        else:
            user_content = user_input

        # Prepare messages for Ollama API
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        
        messages.append({"role": "user", "content": user_content})

        # Map runtime options into Ollama's "options" payload
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        # Call Ollama chat API
        response = self._client.chat(
            model=model,
            messages=messages,
            options=options or None,
        )

        # Extract response text and handle token usage
        text = response.message.content if response.message else ""
        token_usage = {}
        # TODO: Extract real token usage if available from response

        # Return LLMResponse with model_name and token_usage
        return LLMResponse(
            text=text,
            token_usage=token_usage,
            model_name=model,
            raw_output=response,
        )
