from typing import List
from ....interfaces import Document, FileLoader

class TextLoader(FileLoader):
    def can_load(self, path: str) -> bool:
        return path.endswith(".txt")

    def load(self, path: str) -> List[Document]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            
            return [
                Document(
                    id=path,
                    content=content,
                    metadata={"source": path, "type": "text"}
                )
            ]
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return []