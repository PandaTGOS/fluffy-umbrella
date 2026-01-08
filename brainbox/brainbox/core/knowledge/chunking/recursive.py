from typing import List, Optional
import re
from .base import Chunker

class RecursiveChunker(Chunker):
    def __init__(
        self, 
        chunk_size: int = 512, 
        chunk_overlap: int = 64,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Default separators: look for double newline (paragraph), single newline (line), space (word), or empty (char)
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def chunk(self, text: str) -> List[str]:
        return self._split_text(text, self.separators)

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively splits text by the first valid separator."""
        final_chunks = []
        
        # 1. Find the appropriate separator
        separator = separators[-1] # Default to characters
        new_separators = []
        
        for i, sep in enumerate(separators):
            if sep == "": # Character level
                separator = ""
                break
            if re.search(re.escape(sep), text): # If separator exists in text
                separator = sep
                new_separators = separators[i+1:]
                break
                
        # 2. Split
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text) # Char split

        # 3. Merge splits into chunks
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_len = len(split)
            
            # If a single split is too big, recurse on it
            if split_len > self.chunk_size:
                if new_separators:
                    sub_chunks = self._split_text(split, new_separators)
                    final_chunks.extend(sub_chunks)
                else:
                    # Hard chop if no more separators
                    final_chunks.extend(self._hard_chop(split))
                continue

            # Check if adding this split exceeds chunk size
            # Add separator length estimation (naively 1 char or len(sep))
            sep_len = len(separator) if separator else 0
            
            if current_length + split_len + sep_len > self.chunk_size:
                # Flush current chunk
                doc_chunk = self._join_splits(current_chunk, separator)
                if doc_chunk:
                    final_chunks.append(doc_chunk)
                
                # Start new chunk with overlaps if needed (simple implementation for now: just start new)
                # For basic recursive, we often don't strictly enforce overlap on the structural boundaries 
                # as nicely as fixed, but we can keep the last item.
                
                # Improvements: sophisticated overlap logic. 
                # For now: just reset.
                current_chunk = [split]
                current_length = split_len
            else:
                current_chunk.append(split)
                current_length += split_len + sep_len
                
        # Flush remainder
        if current_chunk:
            doc_chunk = self._join_splits(current_chunk, separator)
            if doc_chunk:
                final_chunks.append(doc_chunk)
                
        return final_chunks

    def _join_splits(self, splits: List[str], separator: str) -> str:
        return separator.join(splits).strip()

    def _hard_chop(self, text: str) -> List[str]:
        """Fallback for massive blocks with no separators."""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks
