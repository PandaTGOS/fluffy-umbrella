from langgraph.graph import StateGraph, END
from llama_index.core import Settings
from llama_index.core.llms import ChatMessage, MessageRole

from ...adapters.openai_llm import OpenAILLMProvider

def build_chat_app(llm):
    graph = StateGraph(dict)

    async def chat_node(state):
        # Force the system prompt to be included
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=llm.system_prompt),
            ChatMessage(role=MessageRole.USER, content=state["query"])
        ]
        response = await llm.achat(messages)
        return {"answer": response.message.content}
    

    graph.add_node("chat", chat_node)
    graph.set_entry_point("chat")
    graph.add_edge("chat", END)

    return graph.compile()


class ChatGraphApp:
    def __init__(self, cfg):
        self.cfg = cfg

        llm_provider = OpenAILLMProvider(
            model=cfg.llm.model,
            system_prompt=cfg.llm.system_prompt
        )
        llm_provider.configure()
        
        self.llm = Settings.llm

        self.graph = build_chat_app(self.llm)

    async def arun(self, query: str):
        return await self.graph.ainvoke({"query": query})
