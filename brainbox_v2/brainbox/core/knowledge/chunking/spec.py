from dataclasses import dataclass

@dataclass
class ChunkSpec:
    size: int = 512          # tokens or chars (depending on chunker)
    overlap: int = 64
    strategy: str = "fixed"  # fixed | semantic | sentence
