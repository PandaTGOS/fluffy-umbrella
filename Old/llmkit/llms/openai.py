from ..interfaces import LLMClient, LLMResponse

from typing import Optional, Dict, Any, Generator, Union
from pydantic import BaseModel, ValidationError
from openai import OpenAI
import httpx


class OpenAIClient(LLMClient):

    def __init__(self, default_model: str = "gpt-4.1-nano"):
        self._client = OpenAI(http_client=httpx.Client(verify=False))
        self.default_model = default_model

    # ======================================================================
    # PUBLIC generate()
    # ======================================================================
    def generate(
        self,
        system_instruction: str,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
        output_schema: Optional[type[BaseModel]] = None,
        runtime_options: Optional[Dict[str, Any]] = None,
    ) -> Union[LLMResponse, Generator[str, None, LLMResponse]]:

        runtime_options = runtime_options or {}

        model = runtime_options.get("model", self.default_model)
        stream_enabled = runtime_options.get("stream", False)
        temperature = runtime_options.get("temperature", 0.0)
        max_tokens = runtime_options.get("max_tokens", None)

        # ------------------------------------------------------------------
        # Build user content exactly like OllamaClient does
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Build messages (same structure as Ollama)
        # ------------------------------------------------------------------
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        messages.append({"role": "user", "content": user_content})

        # ------------------------------------------------------------------
        # Build base payload for the OpenAI Responses API
        # ------------------------------------------------------------------
        payload = {
            "model": model,
            "input": messages,           # Responses API key
            "temperature": temperature,
            "stream": stream_enabled,
        }

        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens

        if output_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "schema": output_schema.model_json_schema(),
            }

        # ------------------------------------------------------------------
        # STREAMING MODE
        # ------------------------------------------------------------------
        if stream_enabled:
            return self._streaming_call(payload, output_schema, model)

        # ------------------------------------------------------------------
        # NON-STREAMING MODE
        # ------------------------------------------------------------------
        return self._non_streaming_call(payload, output_schema, model)

    # ======================================================================
    # STREAMING MODE
    # ======================================================================
    def _streaming_call(
        self,
        payload: Dict[str, Any],
        output_schema: Optional[type[BaseModel]],
        model: str
    ) -> Generator[str, None, LLMResponse]:

        collected_tokens: list[str] = []

        with self._client.responses.stream(**payload) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    token = event.delta
                    collected_tokens.append(token)
                    yield token  # send token to caller

            final_response = stream.get_final_response()

        # Flatten into string
        full_text = "".join(collected_tokens)

        # Try to parse JSON schema
        parsed = None
        if output_schema:
            try:
                parsed = output_schema.model_validate_json(full_text)
            except ValidationError:
                parsed = None

        usage = {
            "input_tokens": final_response.usage.input_tokens,
            "output_tokens": final_response.usage.output_tokens,
            "total_tokens": final_response.usage.total_tokens,
        }

        return LLMResponse(
            text=parsed if parsed else full_text,
            token_usage=usage,
            model_name=model,
            raw_output=final_response,
        )

    # ======================================================================
    # NON-STREAMING MODE
    # ======================================================================
    def _non_streaming_call(
        self,
        payload: dict,
        output_schema: Optional[type[BaseModel]],
        model: str
    ) -> LLMResponse:

        response = self._client.responses.create(**payload)

        # ------------------------------------------------------------------
        # Normalize text into SIMPLE STRING always (matches Ollama behavior)
        # ------------------------------------------------------------------
        rt = response.output_text

        if isinstance(rt, list):
            # Responses API can return a list of chunks
            text = "".join(str(chunk) for chunk in rt)
        else:
            text = str(rt)

        # ------------------------------------------------------------------
        # Parse JSON output schema when required
        # ------------------------------------------------------------------
        parsed = None
        if output_schema:
            try:
                parsed = output_schema.model_validate_json(text)
            except ValidationError:
                parsed = None

        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        return LLMResponse(
            text=parsed if parsed else text,
            token_usage=usage,
            model_name=model,
            raw_output=response,
        )
