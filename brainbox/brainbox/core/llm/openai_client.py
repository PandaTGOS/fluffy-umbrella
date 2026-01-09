import os
from typing import Optional, Dict, Any, Union, Iterator
import json
from openai import OpenAI, OpenAIError
from brainbox.core.llm.llm_client import LLMClient, LLMResponse

class OpenAIClient(LLMClient):

    def __init__(
        self, 
        model_name: str = "gpt-4-turbo-preview", 
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_options: Optional[Dict[str, Any]] = None
    ):
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url
        )
        self.model_name = model_name
        self.default_options = default_options or {
            "temperature": 0.0,
            "max_tokens": 1000
        }

    def generate(
        self,
        system_instruction: str,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Any] = None,
        runtime_options: Optional[Dict[str, Any]] = None,
    ) -> Union[LLMResponse, Iterator[str]]:
        
        # Merge options
        options = {**self.default_options, **(runtime_options or {})}
        is_streaming = options.pop("stream", False)
        
        # Build Messages
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_input}
        ]
        
        # Handle Structured Output
        response_format = None
        if output_schema:
            if isinstance(output_schema, dict) and "type" in output_schema:
                 response_format = output_schema
            else:
                 response_format = {"type": "json_object"}
                 if "json" not in system_instruction.lower():
                     messages[0]["content"] += "\n\nIMPORTANT: Return a valid JSON object."

        try:
            if is_streaming:
                return self._generate_stream_internal(
                    model=self.model_name,
                    messages=messages,
                    response_format=response_format,
                    options=options
                )

            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format=response_format,
                **options
            )
            
            message_content = completion.choices[0].message.content
            usage = completion.usage
            
            # Parse JSON if requested
            if response_format and response_format.get("type") == "json_object":
                try:
                    parsed_json = json.loads(message_content)
                    raw_output = parsed_json
                except json.JSONDecodeError:
                    raw_output = {"error": "Failed to parse JSON", "raw": message_content}
            else:
                raw_output = completion.model_dump()

            return LLMResponse(
                text=message_content,
                token_usage={
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens
                },
                model_name=self.model_name,
                raw_output=raw_output
            )

        except OpenAIError as e:
            return LLMResponse(
                text=f"Error generating response: {str(e)}",
                token_usage={},
                model_name=self.model_name,
                raw_output={"error": str(e)}
            )

    def _generate_stream_internal(self, model, messages, response_format, options):
        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                response_format=response_format,
                stream=True,
                **options
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except OpenAIError as e:
            yield f"Error: {str(e)}"
