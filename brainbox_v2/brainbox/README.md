# LLM Enterprise Platform - User Guide

This codebase is a **production-grade, modular, and scalable foundation** for building any type of Large Language Model application. It is designed not just as a set of scripts, but as a flexible framework that adapts to your specific enterprise needs—whether you need a simple chatbot, a high-precision RAG system, or an autonomous multi-agent workforce.

---

## 🏗️ Architecture Philosophy

This platform is built on three core pillars to ensure **Scalability** and **Generalizability**:

1.  **Modular Components (`core/`)**: Every piece of logic (LLM clients, Retrievers, Tools) is a standalone, interchangeable module.
2.  **State-Driven Pipelines (`pipelines/`)**: We use **LangGraph** to model applications as state machines. This allows for complex, non-linear workflows (loops, conditionals, human-in-the-loop) that linear chains cannot handle.
3.  **Strict Typing & Observability**: All inputs/outputs are typed (Pydantic), and every step is observable, ensuring the system is "Enterprise Ready."

---

## 🚀 Capabilities & Use Cases

### 1. Universal RAG (Retrieval Augmented Generation)
*Best for: Knowledge bases, customer support, legal search.*

The platform provides a "Golden" RAG implementation (`apps/golden_rag_app.py`) that goes beyond basic vector search.
- **Hybrid Retrieval**: Combines **Keyword Search** (precision) with **Vector Search** (semantic understanding).
- **Reranking**: Uses a second LLM pass to strictly order documents by relevance.
- **Confidence Gating**: The system *evaluates its own answer*. If retrieval support is low, it refuses to halluncinate.

**How to Scale it:**
- **Add Sources**: Implement a new `Loader` in `core/knowledge/` to ingest PDFs, SQL, or Notion.
- **Change DB**: Swap `InMemoryVectorStore` for `ChromaDB` or `Pinecone` in `core/vectorstore/`.

### 2. Autonomous Multi-Agent Systems
*Best for: Research assistants, coding agents, complex problem solving.*

The platform supports multi-agent coordination (`apps/golden_multi_agent_app.py`).
- **Tool Use**: Agents can wield tools like Calculators, API clients, or Code Executors.
- **Routing**: A "Manager" or "Router" decides which agent or tool is best for the sub-task.
- **Cognitive Loop**: Agents utilize a `Thought -> Action -> Observation` loop (ReAct) to solve problems iteratively.

**How to Scale it:**
- **New Agents**: Subclass `BaseAgent` to create a `WriterAgent` or `ReviewerAgent`.
- **New Tools**: Add python functions to `core/tools/` and register them.

### 3. Structural Extraction & Logic
*Best for: Data processing, form filling, automated tagging.*

By utilizing the `output_schema` parameter in our `LLMClient`, you can force the model to output strict JSON.
- **Use Case**: Turn unstructured emails into structured JSON tickets.

---

## 🛠️ Step-by-Step Implementation Guide

### A. Setting Up Your Environment
1.  **Clone & Install**:
    ```bash
    git clone <repo>
    pip install -r requirements.txt
    ```
2.  **LLM Backend**:
    - By default, we use [Ollama](https://ollama.ai/) for local privacy and zero-cost dev.
    - Run `ollama serve` and `ollama pull llama3`.

### B. Creating Your First App

#### Recipe 1: The "Simple" Chatbot
If you just want to talk to an LLM:

```python
from core.llm import OllamaClient

client = OllamaClient()
response = client.generate(user_input="Hello!")
print(response.text)
```

#### Recipe 2: Adding a Custom Tool
Let's give the LLM the ability to check the weather.

1.  **Create the Tool**:
    ```python
    # core/tools/weather.py
    from core.tools.base import BaseTool

    class WeatherTool(BaseTool):
        name = "get_weather"
        description = "Get current weather for a city."
        
        def run(self, input_data):
            city = input_data.get("city")
            return f"The weather in {city} is Sunny."
    ```

2.  **Register & Use it**:
    ```python
    from core.tools.registry import ToolRegistry
    from core.agents.tool_agent import ToolAgent
    
    registry = ToolRegistry()
    registry.register(WeatherTool())
    
    agent = ToolAgent(client=client, tools=registry)
    result = agent.run("What is the weather in Paris?")
    ```

### C. Swapping Components (Flexibility)

#### Changing the Brain (LLM Provider)
We use a `BaseLLMClient` interface. To use OpenAI or Anthropic:
1.  Check `core/llm/`.
2.  If `openai_client.py` exists, utilize it.
3.  If not, create it by extending `LLMClient`:
    ```python
    class OpenAIClient(LLMClient):
        def generate(self, ...):
            # Call OpenAI API
            pass
    ```

#### Changing the Memory (Vector Store)
To move to production:
1.  Replace `InMemoryVectorStore` with a robust solution.
2.  Implement the `VectorStore` interface (methods: `add`, `search`).

---

## 📂 Codebase Cheat Sheet

| Directory | Purpose | Key Files |
| :--- | :--- | :--- |
| **`apps/`** | **Entry Points**. Start here. | `golden_rag_app.py`, `golden_multi_agent_app.py` |
| **`pipelines/`** | **The Brains**. Logic flow control. | `rag_graph.py` (Flowchart for RAG), `agent_graph.py` (Flowchart for Agents) |
| **`core/llm/`** | **The Engine**. Model interaction. | `ollama_client.py`, `base.py` |
| **`core/tools/`** | **The Hands**. External actions. | `calculator.py`, `registry.py` |
| **`core/knowledge/`** | **The Memory**. Data loading & search. | `retrievers/`, `chunking/` |

---

## 🤝 Contributing & Extending
This platform is a *living template*.
- **Fork it** for your startup or enterprise use case.
- **Modify** `rag_thresholds.py` to tune the strictness of the RAG system.
- **Extend** the `AgentGraph` to add human-approval steps for sensitive actions.

*Built for High-Velocity Engineering Teams.*
