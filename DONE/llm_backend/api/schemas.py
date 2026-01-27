from pydantic import BaseModel
from typing import List, Optional

class Source(BaseModel):
    id: str          # stable identifier
    label: str       # display name
    url: str         # link

class QueryRequest(BaseModel):
    session_id: str
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[Source]] = []