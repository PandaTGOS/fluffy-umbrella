from typing import Dict, Any
from brainbox.core.agents.base import Agent

class AgentRegistry:
    def __init__(self):
        self._agents: Dict[str, Agent] = {}

    def register(self, agent: Agent):
        """Register a new agent."""
        self._agents[agent.name] = agent

    def get(self, name: str) -> Agent:
        """Retrieve an agent by name."""
        if name not in self._agents:
            raise ValueError(f"Agent '{name}' not found in registry.")
        return self._agents[name]
