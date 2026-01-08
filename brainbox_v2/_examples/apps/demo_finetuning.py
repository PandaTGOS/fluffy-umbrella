import os
import json
from brainbox.core.models.fine_tuning.dataset import FineTuneExample, FineTuneDataset
from brainbox.core.models.fine_tuning.ollama_trainer import OllamaFineTuner
from brainbox.core.models.registry import ModelRegistry
from brainbox.core.llm.ollama_client import OllamaClient

def run_finetuning_demo():
    print("--- Phase 1: Dataset Preparation ---")
    data_path = "data/finetuning/dataset.jsonl"
    examples = []
    
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Please run gen_data.py first.")
        return

    with open(data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            examples.append(FineTuneExample(input=data["input"], output=data["output"]))
    
    dataset = FineTuneDataset(examples)
    print(f"Loaded {len(examples)} examples for fine-tuning.")

    print("\n--- Phase 2: Model Training (Modelfile) ---")
    base_model = "deepscaler"
    output_model_name = "deepscaler-cyberpunk-v1"
    
    trainer = OllamaFineTuner()
    model_ref = trainer.train(
        base_model=base_model,
        dataset=dataset,
        output_name=output_model_name
    )
    print(f"Model {model_ref} created successfully.")

    print("\n--- Phase 3: Model Registration ---")
    registry = ModelRegistry()
    registry.register(
        name="cyberpunk_sec",
        model_ref=model_ref,
        metadata={"base": base_model, "domain": "Cyberpunk Security", "version": "1.0"}
    )
    print("Model registered as 'cyberpunk_sec'.")

    print("\n--- Phase 4: Verification (Before vs After) ---")
    query = "Detect the malware signature."
    
    # Before (Base Model)
    client_base = OllamaClient(default_model=base_model)
    print(f"\nQuery: {query}")
    response_base = client_base.generate(
        system_instruction="You are a helpful security assistant.",
        user_input=query
    )
    print(f"Base Model Response (Should be polite/standard):\n{response_base.text}")

    # After (Fine-Tuned Model)
    model_info = registry.get("cyberpunk_sec")
    client_ft = OllamaClient(default_model=model_info["model_ref"])
    response_ft = client_ft.generate(
        system_instruction="",
        user_input=query
    )
    print(f"\nFine-Tuned Model Response (Should have > [SEC_OPS] prefix and technical slang):\n{response_ft.text}")

if __name__ == "__main__":
    run_finetuning_demo()
