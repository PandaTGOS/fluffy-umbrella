import json
import logging
from typing import List, Dict, Any, Optional
from uuid import uuid4

# Conditional import to allow code to be loaded even if deps are missing (for now)
try:
    import asyncpg
except ImportError:
    asyncpg = None

from backend.infrastructure.vector.interface import VectorStore, VectorDocument
from backend.core.config import settings

logger = logging.getLogger(__name__)

class PostgresVectorStore(VectorStore):
    def __init__(self, table_name: str = "embeddings", dim: int = 1536):
        self.pool = None
        self.table_name = table_name
        self.dim = dim
        
    async def connect(self):
        if not asyncpg:
            raise ImportError("asyncpg is required for PostgresVectorStore")
            
        if not self.pool:
            try:
                self.pool = await asyncpg.create_pool(
                    user=settings.POSTGRES_USER,
                    password=settings.POSTGRES_PASSWORD,
                    database=settings.POSTGRES_DB,
                    host=settings.POSTGRES_SERVER,
                    port=settings.POSTGRES_PORT
                )
                await self._init_db()
            except Exception as e:
                logger.error(f"Failed to connect to Postgres: {e}")
                raise

    async def disconnect(self):
        if self.pool:
            await self.pool.close()
            self.pool = None

    async def _init_db(self):
        async with self.pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create table
            # We cast $1 to vector(self.dim) not needed in create table but good to know
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    metadata JSONB,
                    embedding vector({self.dim})
                );
            """)
            
            # Create HNSW index for performance
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx 
                ON {self.table_name} 
                USING hnsw (embedding vector_cosine_ops);
            """)

    async def add_documents(self, documents: List[VectorDocument]) -> None:
        if not self.pool:
            await self.connect()
            
        async with self.pool.acquire() as conn:
            # Register vector type codec if needed, but asyncpg usually handles lists -> array
            # We might need to cast to vector in the query
            
            values = []
            for doc in documents:
                doc_id = doc.id or str(uuid4())
                values.append((
                    doc_id,
                    doc.content,
                    json.dumps(doc.metadata),
                    doc.embedding
                ))
            
            # Batch insert
            await conn.executemany(f"""
                INSERT INTO {self.table_name} (id, content, metadata, embedding)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding;
            """, values)

    async def search(self, query_vector: List[float], limit: int = 5, filters: Dict = None) -> List[VectorDocument]:
        if not self.pool:
            await self.connect()

        # TODO: Implement robust JSONB filtering
        
        # operator <-> is L2 distance, <=> is cosine distance, <#> is negative inner product
        # usually we want cosine distance for embeddings
        query = f"""
            SELECT id, content, metadata, embedding <=> $1 as distance
            FROM {self.table_name}
            ORDER BY distance
            LIMIT $2;
        """
        
        async with self.pool.acquire() as conn:
            # We assume query_vector is a list of floats.
            # Usually asyncpg requires explicit casting to vector if not automatically inferred?
            # Let's stringify it to be safe or rely on asyncpg-pgvector handling if installed.
            # A common safe bet is to pass string '[1,2,3]' and cast ::vector
            
            vector_str = f"[{','.join(map(str, query_vector))}]"
            
            # However, safer to just try passing list and let driver handle or error? 
            # Actually asyncpg doesn't support vector type natively without register_type.
            # Passing string format is safest basic approach without extra setup.
            
            rows = await conn.fetch(query, vector_str, limit)
            
        return [
            VectorDocument(
                id=row['id'],
                content=row['content'],
                metadata=json.loads(row['metadata']),
                embedding=None # Don't return embedding by default to save bandwidth
            ) for row in rows
        ]
