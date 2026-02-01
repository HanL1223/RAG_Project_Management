import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import settings
from src.embeddings import create_embeddings

logger = logging.getLogger(__name__)


#vector store factory
def create_vector_store(
    collection_name: Optional[str] = None,
    persist_directory: Optional[Path | str] = None,
    embeddings: Optional[GoogleGenerativeAIEmbeddings] = None,
) -> Chroma:
    
    # Use settings defaults if not provided
    collection_name = collection_name or settings.chroma.collection
    persist_directory = persist_directory or settings.chroma.persist_dir

    persist_directory = str(Path(persist_directory).resolve())

    #Create embedding if not provied
    if embeddings is None:
        embeddings - create_embeddings()
    logger.debug(
        f"Creating Chroma vector store: "
        f"collection={collection_name}, "
        f"persist_dir={persist_directory}"
    )

    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """
    Get a cached singleton vector store instance.
    
    WHY CACHE?
    Vector stores maintain connections and state:
      - Database connections
      - In-memory indexes
      - Embedding function reference
    
    Caching avoids reconnecting on each request.
    
    Returns:
        Cached Chroma vector store instance
    """
    return create_vector_store()

def clear_cache():
    """Clear the cached vector store instance."""
    get_vector_store.cache_clear()



#Document operation


def add_documents(
        documents:list[Document],
        ids:Optional[list[str]] = None,
        collection_name:Optional[str] = None,
)-> list[str]:
    if not documents:
        logging.warning('No document to add')
        return []
    if collection_name:
        store = create_vector_store(collection_name=collection_name)
    else:
        store = get_vector_store()

    result_ids = store.add_documents(documents,ids = ids)
    logger.info(f"Added {len(documents)} documents to vector store")
    return result_ids

def create_from_documents(
    documents: list[Document],
    ids: Optional[list[str]] = None,
    collection_name: Optional[str] = None,
    persist_directory: Optional[Path | str] = None,
) -> Chroma:
    collection_name = collection_name or settings.chroma.collection
    persist_directory = str(persist_directory or settings.chroma.persist_dir)
    embeddings = create_embeddings()
    
    logger.info(
        f"Creating new vector store with {len(documents)} documents "
        f"in collection '{collection_name}'"
    )
    
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        ids=ids,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )


#Search Opeerations
def similarity_search(
    query: str,
    k: Optional[int] = None,
    filter_dict: Optional[dict[str, Any]] = None,
) -> list[Document]:
    """
    Search for similar documents.
    
    SIMILARITY SEARCH:
    1. Query text is embedded
    2. Embedding is compared to stored vectors
    3. Top-k most similar documents returned
    
    FILTERING:
    ChromaDB supports metadata filtering:
      - {"status": "Done"} - Exact match
      - {"priority": {"$in": ["High", "Critical"]}} - In list
      - {"created": {"$gte": "2024-01-01"}} - Comparison
    
    Args:
        query: Search query text
        k: Number of results (default from settings)
        filter_dict: Optional metadata filter
        
    Returns:
        List of similar documents (most similar first)
        
    Example:
        >>> docs = similarity_search("data modeling", k=5)
        >>> docs[0].page_content  # Most similar
    """
    k = k or settings.rag.top_k
    store = get_vector_store()
    
    if filter_dict:
        return store.similarity_search(query, k=k, filter=filter_dict)
    return store.similarity_search(query, k=k)

def similarity_search_with_score(
    query: str,
    k: Optional[int] = None,
    filter_dict: Optional[dict[str, Any]] = None,
) -> list[tuple[Document, float]]:
    """
    Search with similarity scores.
    
    SCORES:
    The score is the distance metric (lower = more similar for L2).
    For cosine distance: score 0 = identical, score 2 = opposite.
    
    Args:
        query: Search query text
        k: Number of results
        filter_dict: Optional metadata filter
        
    Returns:
        List of (document, score) tuples
        
    Example:
        >>> results = similarity_search_with_score("data modeling")
        >>> for doc, score in results:
        ...     print(f"{doc.metadata['issue_key']}: {score:.4f}")
    """
    k = k or settings.rag.top_k
    store = get_vector_store()
    
    if filter_dict:
        return store.similarity_search_with_score(query, k=k, filter=filter_dict)
    return store.similarity_search_with_score(query, k=k)



def get_collection_count() -> int:
    """
    Get the number of documents in the collection.
    
    Returns:
        Number of documents (0 if collection doesn't exist)
    """
    try:
        store = get_vector_store()
        # Access the underlying ChromaDB collection
        return store._collection.count()
    except Exception as e:
        logger.warning(f"Failed to get collection count: {e}")
        return 0


def collection_exists() -> bool:
    """
    Check if the collection exists and has documents.
    
    Returns:
        True if collection exists with at least one document
    """
    return get_collection_count() > 0


def delete_collection(collection_name: Optional[str] = None) -> bool:
    """
    Delete a collection (for testing/cleanup).
    
    WARNING: This permanently deletes all data in the collection!
    
    Args:
        collection_name: Collection to delete (default from settings)
        
    Returns:
        True if deletion succeeded
    """
    collection_name = collection_name or settings.chroma.collection
    
    try:
        import chromadb
        
        persist_dir = str(settings.chroma.persist_dir)
        client = chromadb.PersistentClient(path=persist_dir)
        client.delete_collection(collection_name)
        
        # Clear cache since collection is gone
        clear_cache()
        
        logger.info(f"Deleted collection: {collection_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to delete collection: {e}")
        return False