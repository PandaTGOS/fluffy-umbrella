from typing import Any, List, Optional, AsyncGenerator, Protocol
from pydantic import BaseModel

# We define our own message types to decouple the *API* from LangChain internals
# But internally we will map these to LangChain messages
class LLMMessage(BaseModel):
    role: str
    content: Optional[Any] # support for multimodal
    name: Optional[str] = None
    tool_calls: Optional[list] = None
    tool_call_id: Optional[str] = None

class LLMResponse(BaseModel):
    content: str
    tool_calls: Optional[list] = None
    usage: Optional[dict] = None

class LLMProvider(Protocol):
    """
    Protocol that all LLM adapters must follow.
    This allows us to swap out LangChain for something else entirely if needed,
    though practically we will implement this using LangChain components.
    """
    
    async def generate(
        self, 
        messages: List[LLMMessage], 
        tools: Optional[List[Any]] = None,
        **kwargs
    ) -> LLMResponse:
        ...

    async def stream(
        self, 
        messages: List[LLMMessage], 
        tools: Optional[List[Any]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        ...
