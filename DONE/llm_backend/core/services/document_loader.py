import os
from llama_index.core import SimpleDirectoryReader

def extract_metadata(path: str):
    path_lower = path.lower()
    return {
        "language": "en" if "english" in path_lower else "sv",
        "category": os.path.basename(os.path.dirname(path)),
        "filename": os.path.basename(path)
    }

def load_documents(data_dir: str):
    
    loader = SimpleDirectoryReader(
        input_dir=data_dir,
        recursive=True,
        file_metadata=extract_metadata
    )

    return loader.load_data()
