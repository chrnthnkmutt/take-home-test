"""
Data ingestion pipeline for the AI Product Research Assistant.

This module handles loading product data from CSV, generating embeddings,
and storing them in the ChromaDB vector database.
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from dotenv import load_dotenv

from src.pipeline.vector_store import VectorStore, get_vector_store

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings using Azure OpenAI or Sentence Transformers as fallback.
    """
    
    def __init__(self, use_azure: bool = True):
        """
        Initialize the embedding generator.
        
        Args:
            use_azure: Whether to use Azure OpenAI embeddings (requires AZURE_OPENAI_API_KEY)
        """
        self.use_azure = use_azure and os.getenv("AZURE_OPENAI_API_KEY") is not None
        
        if self.use_azure:
            try:
                from langchain_openai import AzureOpenAIEmbeddings
                self.client = AzureOpenAIEmbeddings(
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"),
                    api_version="2023-05-15"
                )
                self.model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
                logger.info(f"Using Azure OpenAI embeddings ({self.model})")
            except Exception as e:
                logger.warning(f"Failed to initialize Azure OpenAI client: {e}. Falling back to sentence-transformers")
                self.use_azure = False
        
        if not self.use_azure:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Using sentence-transformers embeddings (all-MiniLM-L6-v2)")
            except Exception as e:
                logger.error(f"Failed to initialize sentence-transformers: {e}")
                raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            List of embedding vectors
        """
        try:
            if self.use_azure:
                return self._generate_azure_embeddings(texts)
            else:
                return self._generate_sentence_transformer_embeddings(texts)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def _generate_azure_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using Azure OpenAI API.
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            List of embedding vectors
        """
        embeddings = []
        batch_size = 100  # Process in batches for better performance
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                batch_embeddings = self.client.embed_documents(batch)
                embeddings.extend(batch_embeddings)
                logger.info(f"Generated Azure OpenAI embeddings for batch {i//batch_size + 1} ({len(batch)} texts)")
            except Exception as e:
                logger.error(f"Error generating Azure OpenAI embeddings for batch {i//batch_size + 1}: {e}")
                raise
        
        return embeddings
    
    def _generate_sentence_transformer_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using Sentence Transformers.
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.model.encode(texts, show_progress_bar=True)
            logger.info(f"Generated sentence-transformer embeddings for {len(texts)} texts")
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating sentence-transformer embeddings: {e}")
            raise


def load_products_csv(csv_path: str) -> pd.DataFrame:
    """
    Load and parse the products catalog CSV file.
    
    Args:
        csv_path: Path to the CSV file
    
    Returns:
        DataFrame containing product data
    """
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} products from {csv_path}")
        
        # Validate required columns
        required_columns = [
            'product_id', 'product_name', 'category', 'brand', 
            'description', 'current_price', 'cost', 'stock_quantity', 
            'average_rating'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        raise


def create_product_text(row: pd.Series) -> str:
    """
    Create a combined text representation of a product for embedding.
    
    Args:
        row: DataFrame row containing product data
    
    Returns:
        Combined text string
    """
    # Combine name, description, category, and brand into a single text
    text_parts = [
        f"Product: {row['product_name']}",
        f"Category: {row['category']}",
        f"Brand: {row['brand']}",
        f"Description: {row['description']}"
    ]
    
    return " | ".join(text_parts)


def chunk_product_description(description: str, max_length: int = 500) -> List[str]:
    """
    Chunk long product descriptions if needed.
    
    For most product descriptions, chunking is not necessary as they are typically
    short. This function is provided for completeness but may not be used in practice.
    
    Args:
        description: Product description text
        max_length: Maximum length of each chunk
    
    Returns:
        List of text chunks
    """
    if len(description) <= max_length:
        return [description]
    
    # Simple chunking by sentences
    sentences = description.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length > max_length and current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    if current_chunk:
        chunks.append('. '.join(current_chunk))
    
    return chunks


def prepare_product_data(df: pd.DataFrame) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """
    Prepare product data for ingestion into vector store.
    
    Args:
        df: DataFrame containing product data
    
    Returns:
        Tuple of (documents, metadatas, ids)
    """
    documents = []
    metadatas = []
    ids = []
    
    for _, row in df.iterrows():
        # Create product text for embedding
        product_text = create_product_text(row)
        documents.append(product_text)
        
        # Determine stock status
        stock_quantity = row.get('stock_quantity', 0)
        if stock_quantity > 100:
            stock_status = "in_stock"
        elif stock_quantity > 0:
            stock_status = "low_stock"
        else:
            stock_status = "out_of_stock"
        
        # Create metadata dictionary
        metadata = {
            "product_id": str(row['product_id']),
            "name": str(row['product_name']),
            "category": str(row['category']),
            "brand": str(row['brand']),
            "price": float(row['current_price']),
            "cost": float(row['cost']),
            "stock_status": stock_status,
            "stock_quantity": int(stock_quantity),
            "rating": float(row['average_rating'])
        }
        
        metadatas.append(metadata)
        ids.append(str(row['product_id']))
    
    logger.info(f"Prepared {len(documents)} products for ingestion")
    return documents, metadatas, ids


def ingest_products(
    csv_path: str = "data/products_catalog.csv",
    persist_directory: str = "./chroma_db",
    collection_name: str = "products",
    use_azure: bool = True,
    batch_size: int = 100
) -> Dict[str, Any]:
    """
    Main function to orchestrate the product ingestion pipeline.
    
    This function:
    1. Loads product data from CSV
    2. Prepares product texts and metadata
    3. Generates embeddings
    4. Stores embeddings in ChromaDB vector store
    
    The function is idempotent - it uses upsert operations so running it
    multiple times will update existing products and add new ones.
    
    Args:
        csv_path: Path to the products CSV file
        persist_directory: Directory to persist ChromaDB data
        collection_name: Name of the ChromaDB collection
        use_azure: Whether to use Azure OpenAI embeddings (requires AZURE_OPENAI_API_KEY)
        batch_size: Number of products to process in each batch
    
    Returns:
        Dictionary with ingestion statistics
    """
    try:
        logger.info("Starting product ingestion pipeline")
        
        # Load product data
        df = load_products_csv(csv_path)
        total_products = len(df)
        
        # Prepare product data
        documents, metadatas, ids = prepare_product_data(df)
        
        # Initialize embedding generator
        embedding_generator = EmbeddingGenerator(use_azure=use_azure)
        
        # Initialize vector store
        vector_store = get_vector_store(
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        vector_store.get_or_create_collection()
        
        # Process in batches
        ingested_count = 0
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_metas = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            
            logger.info(f"Processing batch {i//batch_size + 1} ({len(batch_docs)} products)")
            
            # Generate embeddings
            embeddings = embedding_generator.generate_embeddings(batch_docs)
            
            # Store in vector database
            vector_store.add_documents(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids,
                embeddings=embeddings
            )
            
            ingested_count += len(batch_docs)
            logger.info(f"Ingested {ingested_count}/{total_products} products")
        
        # Get final statistics
        stats = vector_store.get_collection_stats()
        stats['ingested_products'] = ingested_count
        stats['total_products'] = total_products
        stats['embedding_model'] = 'azure-openai' if embedding_generator.use_azure else 'sentence-transformers'
        
        logger.info(f"Product ingestion completed successfully: {stats}")
        return stats
    
    except Exception as e:
        logger.error(f"Error in product ingestion pipeline: {e}")
        raise


def update_products(
    product_ids: List[str],
    csv_path: str = "data/products_catalog.csv",
    persist_directory: str = "./chroma_db",
    collection_name: str = "products",
    use_azure: bool = True
) -> Dict[str, Any]:
    """
    Update specific products in the vector store (incremental update).
    
    Args:
        product_ids: List of product IDs to update
        csv_path: Path to the products CSV file
        persist_directory: Directory to persist ChromaDB data
        collection_name: Name of the ChromaDB collection
        use_azure: Whether to use Azure OpenAI embeddings
    
    Returns:
        Dictionary with update statistics
    """
    try:
        logger.info(f"Starting incremental update for {len(product_ids)} products")
        
        # Load product data
        df = load_products_csv(csv_path)
        
        # Filter to only the products we want to update
        df_filtered = df[df['product_id'].isin(product_ids)]
        
        if len(df_filtered) == 0:
            logger.warning(f"No products found matching IDs: {product_ids}")
            return {"updated_products": 0, "message": "No matching products found"}
        
        # Prepare product data
        documents, metadatas, ids = prepare_product_data(df_filtered)
        
        # Initialize embedding generator
        embedding_generator = EmbeddingGenerator(use_azure=use_azure)
        
        # Generate embeddings
        embeddings = embedding_generator.generate_embeddings(documents)
        
        # Initialize vector store
        vector_store = get_vector_store(
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        vector_store.get_or_create_collection()
        
        # Update in vector database (upsert will update existing or insert new)
        vector_store.add_documents(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        
        stats = {
            "updated_products": len(documents),
            "product_ids": ids,
            "embedding_model": 'azure-openai' if embedding_generator.use_azure else 'sentence-transformers'
        }
        
        logger.info(f"Product update completed successfully: {stats}")
        return stats
    
    except Exception as e:
        logger.error(f"Error in product update: {e}")
        raise


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the ingestion pipeline
    try:
        stats = ingest_products()
        print(f"\nIngestion completed successfully!")
        print(f"Statistics: {stats}")
    except Exception as e:
        print(f"\nIngestion failed: {e}")
        raise


