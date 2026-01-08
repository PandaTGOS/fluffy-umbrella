from typing import List
from .base import Chunker

class FixedChunker(Chunker):
    def __init__(self, size: int = 512, overlap: int = 64):
        self.size = size
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        if not text:
            return []
            
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + self.size
            # Clamp end to length
            if end > text_len:
                end = text_len
                
            chunk = text[start:end]
            chunks.append(chunk)
            
            # Stop if we reached the end
            if end == text_len:
                break
                
            # Move start forward by stride (size - overlap)
            # Ensure we progress even if overlap >= size (sanity check)
            stride = max(1, self.size - self.overlap)
            start += stride
            
        return chunks
