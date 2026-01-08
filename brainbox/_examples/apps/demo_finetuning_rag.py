from brainbox.core.llm.ollama_client import OllamaClient
from brainbox.core.models.registry import ModelRegistry
import os

def mock_retrieve(query: str):
    if "malware" in query.lower():
        return "Log Entry: Unauthorized access detected at 04:00 UTC. IP 192.168.1.45 attempted buffer overflow on kernel node."
    return "No relevant logs found."

def run_finetuning_rag_demo():
    print("--- Cyberpunk Fine-Tuning + RAG Demo ---")
    
    ft_model_name = "deepscaler-cyberpunk-v1"
    print(f"Using Fine-Tuned Model: {ft_model_name}")
    
    client = OllamaClient(default_model=ft_model_name)
    
    query = "Analyze the traffic spike and check for malware signatures in current logs."
    print(f"Query: {query}")
    
    context = mock_retrieve(query)
    print(f"Retrieved Context: {context}")
    
    response = client.generate(
        system_instruction="",
        user_input=query,
        context={"logs": context}
    )
    
    print("\n--- Final Output (Should show fine-tuned tone + RAG data) ---")
    print(response.text)

if __name__ == "__main__":
    run_finetuning_rag_demo()
