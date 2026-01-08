import ollama
from .trainer import FineTuner
from .dataset import FineTuneDataset
import os

class OllamaFineTuner(FineTuner):
    def __init__(self, host: str = None):
        self.client = ollama.Client(host=host)

    def train(
        self,
        base_model: str,
        dataset: FineTuneDataset,
        output_name: str
    ) -> str:
        """
        In Ollama context, 'training' or 'fine-tuning' often refers to creating 
        a new model with a specialized system prompt or template derived from 
        the dataset, as true weight-based fine-tuning happens externally.
        
        This implementation creates a new Ollama model with a specialized 
        system instruction based on the dataset examples to guide behavior.
        """
        # Create a combined system instruction from the dataset examples
        # to inject 'demonstrations' into the model's base behavior.
        examples_str = "\n".join([
            f"Input: {ex.input}\nOutput: {ex.output}" 
            for ex in dataset.examples[:5] # Use top 5 as demonstrations
        ])
        
        # Use ollama to create the model with a specialized system prompt
        print(f"Creating Ollama model: {output_name} from {base_model}...")
        self.client.create(
            model=output_name, 
            from_=base_model,
            system=f"You are a specialized agent. Below are examples of how you should respond:\n{examples_str}\nMaintain this tone and structure for all future interactions."
        )
        
        return output_name
