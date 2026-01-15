from typing import List, Optional
import re
from ..interfaces import Chunker

class RecursiveChunker(Chunker):
    def __init__(
        self, 
        chunk_size: int = 512, 
        chunk_overlap: int = 64,
        separators: Optional[List[str]] = None,
        keep_headers: bool = True
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Default separators: look for double newline (paragraph), single newline (line), space (word), or empty (char)
        self.separators = separators or ["\n\n", "\n", " ", ""]
        self.keep_headers = keep_headers

    def chunk(self, text: str) -> List[str]:
        # If we want to respect headers, we should split by headers first
        if self.keep_headers:
            return self._chunk_with_headers(text)
        return self._split_text(text, self.separators)

    def _chunk_with_headers(self, text: str) -> List[str]:
        """
        Splits text by markdown headers and chunks content within sections, 
        prepending the header path to each chunk.
        """
        lines = text.split('\n')
        sections = []
        current_headers = []
        current_lines = []
        
        # Simple parser to identify sections
        for line in lines:
            header_match = re.match(r"^(#{1,6})\s+(.*)", line)
            if header_match:
                # Flush current section
                if current_lines:
                    sections.append({
                        "headers": list(current_headers),
                        "text": "\n".join(current_lines)
                    })
                    current_lines = []
                
                # Update headers logic
                # If new header level is N, remove all headers >= N
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                
                # We need to maintain a stack of headers. 
                # This simple stack assumes well-formed markdown where headers are nested.
                # If we encounter a header level L, we pop everything that is >= L.
                # However, since we don't track levels in the simple list, let's track (level, title) tuples.
                
                # Clean up stack
                while current_headers and current_headers[-1][0] >= level:
                    current_headers.pop()
                
                current_headers.append((level, title))
            else:
                current_lines.append(line)
        
        # Flush last section
        if current_lines:
            sections.append({
                "headers": list(current_headers),
                "text": "\n".join(current_lines)
            })

        # Process sections into chunks
        final_chunks = []
        for section in sections:
            header_context = " > ".join([h[1] for h in section["headers"]])
            section_text = section["text"]
            
            # Chunk the section text
            raw_chunks = self._split_text(section_text, self.separators)
            
            for rc in raw_chunks:
                if header_context:
                    # Prepend context
                    final_chunks.append(f"Context: {header_context}\n\n{rc}")
                else:
                    final_chunks.append(rc)
                    
        return final_chunks

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
                
                # Start new chunk
                # Ideally, we should support overlap here properly by shifting a window
                # But for this recursive implementation, let's keep it simple as it was before
                # To enable overlap we would need to look back at previous splits.
                # EXISTING CODE did not seem to implement sliding window overlap well, 
                # "chunk_overlap" var was unused in main loop logic except for hard_chop.
                
                # Let's try to add overlap if possible? 
                # For now, let's stick to base implementation of logic to avoid regression, 
                # just wrapper for headers.
                
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