import os
from typing import List
from .base import FileLoader
from brainbox.core.knowledge import Document

class DirectoryLoader:
    def __init__(self, loaders: List[FileLoader]):
        self.loaders = loaders

    def load(self, directory: str) -> List[Document]:
        documents = []
        
        if not os.path.isdir(directory):
            print(f"[WARN] Directory not found: {directory}")
            return []

        for root, _, files in os.walk(directory):
            for file in files:
                path = os.path.join(root, file)

                for loader in self.loaders:
                    if loader.can_load(path):
                        docs = loader.load(path)
                        documents.extend(docs)
                        break 
                        # Stop checking loaders once one has claimed the file
        
        return documents
