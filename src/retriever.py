import os
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from src.embeddings import init_settings

# Initialize settings
init_settings()

def get_index(documents=None):
    """
    Creates an In-Memory Vector Index.
    """
    # ---------------------------------------------------------
    # THE FIX: Use ":memory:" instead of a file path.
    # This prevents the "Storage folder already accessed" error.
    # ---------------------------------------------------------
    client = QdrantClient(location=":memory:")
    
    vector_store = QdrantVectorStore(client=client, collection_name="research_papers")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Always build from documents (since memory is wiped on restart)
    if documents:
        return VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context
        )
    else:
        # Return an empty index if no docs yet
        return VectorStoreIndex.from_vector_store(vector_store=vector_store)

def get_query_engine(index):
    # We now pass the index directly to this function
    return index.as_query_engine(similarity_top_k=5)