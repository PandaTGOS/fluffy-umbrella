from abc import ABC, abstractmethod

class SessionMemory(ABC):
    @abstractmethod
    def get_summary(self, session_id: str) -> str:
        ...

    @abstractmethod
    def update(
        self,
        session_id: str,
        user_query: str,
        assistant_answer: str,
        llm  
    ) -> None:
        ...
