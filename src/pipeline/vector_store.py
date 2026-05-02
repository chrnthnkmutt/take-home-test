"""
Vector store management using ChromaDB for the AI Product Research Assistant.

This module provides functions for managing the ChromaDB vector database,
including collection creation, querying, and metadata filtering.
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages ChromaDB vector store operations for product embeddings.
    
    Attributes:
        client: ChromaDB client instance
        collection_name: Name of the collection to use
        embedding_function: Function to generate embeddings
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "products",
        embedding_function: Optional[Any] = None
    ):
        """
        Initialize the VectorStore with ChromaDB client.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection to create/use
            embedding_function: Optional custom embedding function
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.query_embedding_model = self._create_query_embedding_model()
        
        # Initialize ChromaDB client with persistent storage
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        self.embedding_function = embedding_function
        self.collection = None
        
        logger.info(f"VectorStore initialized with persist_directory: {persist_directory}")

    def _create_query_embedding_model(self) -> Optional[Any]:
        """
        Create the embedding model used for query-time embeddings.

        Returns:
            AzureOpenAIEmbeddings or SentenceTransformer instance, or None if unavailable.
        """
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")

        if azure_api_key and azure_endpoint:
            try:
                from langchain_openai import AzureOpenAIEmbeddings

                logger.info(f"Using Azure OpenAI query embeddings ({azure_deployment})")
                return AzureOpenAIEmbeddings(
                    azure_endpoint=azure_endpoint,
                    api_key=azure_api_key,
                    azure_deployment=azure_deployment,
                    api_version="2023-05-15"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Azure query embeddings: {e}")

        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Using sentence-transformers query embeddings (all-MiniLM-L6-v2)")
            return SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning(f"Failed to initialize sentence-transformers query embeddings: {e}")
            return None

    def _embed_query_texts(self, query_texts: List[str]) -> Optional[List[List[float]]]:
        """
        Generate embeddings for query texts when a matching embedding model is available.

        Args:
            query_texts: List of query strings

        Returns:
            Query embeddings or None if no model is available.
        """
        if not self.query_embedding_model:
            return None

        try:
            if hasattr(self.query_embedding_model, "embed_query"):
                return [self.query_embedding_model.embed_query(text) for text in query_texts]

            if hasattr(self.query_embedding_model, "encode"):
                embeddings = self.query_embedding_model.encode(query_texts, show_progress_bar=False)
                return embeddings.tolist()

        except Exception as e:
            logger.warning(f"Failed to embed query texts: {e}")

        return None
    
    def get_or_create_collection(self) -> chromadb.Collection:
        """
        Get existing collection or create a new one.
        
        Returns:
            ChromaDB collection instance
        """
        try:
            if self.embedding_function:
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": "cosine"}
                )
            else:
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            
            logger.info(f"Collection '{self.collection_name}' ready with {self.collection.count()} documents")
            return self.collection
        
        except Exception as e:
            logger.error(f"Error getting/creating collection: {e}")
            raise
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None
    ) -> None:
        """
        Add documents to the collection with metadata.
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries
            ids: List of unique document IDs
            embeddings: Optional pre-computed embeddings
        """
        if not self.collection:
            self.get_or_create_collection()
        
        try:
            if embeddings:
                self.collection.upsert(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings
                )
            else:
                self.collection.upsert(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            
            logger.info(f"Added/updated {len(documents)} documents to collection")
        
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def query(
        self,
        query_texts: List[str],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the vector store with optional metadata filters.
        
        Args:
            query_texts: List of query texts
            n_results: Number of results to return
            where: Metadata filter conditions (e.g., {"category": "Electronics"})
            where_document: Document content filter conditions
        
        Returns:
            Dictionary containing query results with ids, documents, metadatas, and distances
        """
        if not self.collection:
            self.get_or_create_collection()
        
        try:
            query_embeddings = self._embed_query_texts(query_texts)

            if query_embeddings is not None:
                results = self.collection.query(
                    query_embeddings=query_embeddings,
                    n_results=n_results,
                    where=where,
                    where_document=where_document
                )
            else:
                results = self.collection.query(
                    query_texts=query_texts,
                    n_results=n_results,
                    where=where,
                    where_document=where_document
                )
            
            logger.info(f"Query returned {len(results['ids'][0])} results")
            return results
        
        except Exception as e:
            logger.error(f"Error querying collection: {e}")
            raise
    
    def query_with_filters(
        self,
        query_text: str,
        n_results: int = 10,
        category: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        stock_status: Optional[str] = None,
        brand: Optional[str] = None,
        min_rating: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Query with common product filters.
        
        Args:
            query_text: Search query text
            n_results: Number of results to return
            category: Filter by product category
            min_price: Minimum price filter
            max_price: Maximum price filter
            stock_status: Filter by stock status
            brand: Filter by brand
            min_rating: Minimum rating filter
        
        Returns:
            Dictionary containing filtered query results
        """
        where_filter = {}
        
        if category:
            where_filter["category"] = category
        
        if stock_status:
            where_filter["stock_status"] = stock_status
        
        if brand:
            where_filter["brand"] = brand
        
        # ChromaDB supports $gte, $lte operators for numeric comparisons
        if min_price is not None:
            where_filter["price"] = {"$gte": min_price}
        
        if max_price is not None:
            if "price" in where_filter:
                where_filter["price"]["$lte"] = max_price
            else:
                where_filter["price"] = {"$lte": max_price}
        
        if min_rating is not None:
            where_filter["rating"] = {"$gte": min_rating}
        
        return self.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter if where_filter else None
        )
    
    def is_initialized(self) -> bool:
        """
        Check if the vector store is initialized with data.
        
        Returns:
            True if collection exists and has documents, False otherwise
        """
        try:
            if not self.collection:
                self.get_or_create_collection()
            
            count = self.collection.count()
            return count > 0
        
        except Exception as e:
            logger.error(f"Error checking initialization: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        if not self.collection:
            self.get_or_create_collection()
        
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory
            }
        
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {
                "collection_name": self.collection_name,
                "document_count": 0,
                "persist_directory": self.persist_directory,
                "error": str(e)
            }
    
    def delete_collection(self) -> None:
        """
        Delete the collection from ChromaDB.
        Warning: This will remove all stored embeddings.
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = None
            logger.info(f"Collection '{self.collection_name}' deleted")
        
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise


def get_vector_store(
    persist_directory: str = "./chroma_db",
    collection_name: str = "products",
    embedding_function: Optional[Any] = None
) -> VectorStore:
    """
    Factory function to create and return a VectorStore instance.
    
    Args:
        persist_directory: Directory to persist ChromaDB data
        collection_name: Name of the collection to create/use
        embedding_function: Optional custom embedding function
    
    Returns:
        VectorStore instance
    """
    return VectorStore(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embedding_function
    )


