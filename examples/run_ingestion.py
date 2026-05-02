"""
Example script demonstrating how to use the data ingestion pipeline.

This script shows how to:
1. Run the full ingestion pipeline
2. Query the vector store
3. Update specific products
"""

import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline.ingestion import ingest_products, update_products
from src.pipeline.vector_store import get_vector_store

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def run_full_ingestion():
    """Run the complete ingestion pipeline."""
    logger.info("=" * 60)
    logger.info("Running Full Product Ingestion")
    logger.info("=" * 60)
    
    try:
        stats = ingest_products(
            csv_path="data/products_catalog.csv",
            persist_directory="./chroma_db",
            collection_name="products",
            use_azure=True,  # Set to False to use sentence-transformers
            batch_size=100
        )
        
        logger.info("\nIngestion completed successfully!")
        logger.info(f"Total products: {stats['total_products']}")
        logger.info(f"Ingested products: {stats['ingested_products']}")
        logger.info(f"Embedding model: {stats['embedding_model']}")
        logger.info(f"Collection: {stats['collection_name']}")
        logger.info(f"Document count: {stats['document_count']}")
        
        return stats
    
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise


def test_queries():
    """Test querying the vector store."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Vector Store Queries")
    logger.info("=" * 60)
    
    try:
        # Initialize vector store
        vector_store = get_vector_store()
        vector_store.get_or_create_collection()
        
        # Test 1: Basic query
        logger.info("\n--- Test 1: Basic Query ---")
        results = vector_store.query(
            query_texts=["wireless headphones with noise cancellation"],
            n_results=3
        )
        
        logger.info(f"Found {len(results['ids'][0])} results:")
        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            similarity = 1 - distance
            logger.info(f"  {i+1}. {metadata['name']}")
            logger.info(f"     Price: ${metadata['price']}, Rating: {metadata['rating']}, Similarity: {similarity:.3f}")
        
        # Test 2: Query with filters
        logger.info("\n--- Test 2: Query with Filters ---")
        results = vector_store.query_with_filters(
            query_text="fitness equipment",
            n_results=3,
            category="Sports & Fitness",
            max_price=50.0,
            min_rating=4.5
        )
        
        logger.info(f"Found {len(results['ids'][0])} results (Sports & Fitness, <$50, rating>=4.5):")
        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]
            logger.info(f"  {i+1}. {metadata['name']}")
            logger.info(f"     Price: ${metadata['price']}, Rating: {metadata['rating']}")
        
        # Test 3: Category-specific query
        logger.info("\n--- Test 3: Category-Specific Query ---")
        results = vector_store.query_with_filters(
            query_text="kitchen appliances",
            n_results=3,
            category="Kitchen & Dining"
        )
        
        logger.info(f"Found {len(results['ids'][0])} results (Kitchen & Dining):")
        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]
            logger.info(f"  {i+1}. {metadata['name']}")
            logger.info(f"     Price: ${metadata['price']}, Brand: {metadata['brand']}")
        
        logger.info("\nAll queries completed successfully!")
        
    except Exception as e:
        logger.error(f"Query testing failed: {e}")
        raise


def run_incremental_update():
    """Demonstrate incremental product updates."""
    logger.info("\n" + "=" * 60)
    logger.info("Running Incremental Product Update")
    logger.info("=" * 60)
    
    try:
        # Update specific products
        product_ids = ["PROD-001", "PROD-002", "PROD-003"]
        
        logger.info(f"Updating products: {product_ids}")
        
        stats = update_products(
            product_ids=product_ids,
            csv_path="data/products_catalog.csv"
        )
        
        logger.info("\nUpdate completed successfully!")
        logger.info(f"Updated products: {stats['updated_products']}")
        logger.info(f"Product IDs: {stats['product_ids']}")
        logger.info(f"Embedding model: {stats['embedding_model']}")
        
        return stats
    
    except Exception as e:
        logger.error(f"Update failed: {e}")
        raise


def main():
    """Main function to run all examples."""
    logger.info("AI Product Research Assistant - Ingestion Pipeline Examples")
    logger.info("=" * 60)
    
    try:
        # Step 1: Run full ingestion
        ingestion_stats = run_full_ingestion()
        
        # Step 2: Test queries
        test_queries()
        
        # Step 3: Demonstrate incremental update (optional)
        # Uncomment to test incremental updates
        # update_stats = run_incremental_update()
        
        logger.info("\n" + "=" * 60)
        logger.info("All examples completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"\nExample execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


