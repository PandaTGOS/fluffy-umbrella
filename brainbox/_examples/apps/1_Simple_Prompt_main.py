from brainbox.pipelines import SimplePipeline
from brainbox.core.llm import OllamaClient

# Setup
client = OllamaClient()
pipeline = SimplePipeline(client=client)

# Run
question = "How to make concentrated dark matter"
result = pipeline.run(question)

# Output
print(f"Model Name: {result['model_name']}")
print(f"Response Text: {result['answer']}")
print(f"Token Usage: {result['token_usage']}")