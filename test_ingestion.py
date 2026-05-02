"""
Simple test script to verify the ingestion pipeline implementation.
"""

import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        from src.pipeline.vector_store import VectorStore, get_vector_store
        logger.info("✓ vector_store module imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import vector_store: {e}")
        return False
    
    try:
        from src.pipeline.ingestion import (
            EmbeddingGenerator,
            load_products_csv,
            create_product_text,
            prepare_product_data,
            ingest_products
        )
        logger.info("✓ ingestion module imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import ingestion: {e}")
        return False
    
    return True

def test_csv_loading():
    """Test loading the CSV file."""
    logger.info("Testing CSV loading...")
    
    try:
        from src.pipeline.ingestion import load_products_csv
        df = load_products_csv("data/products_catalog.csv")
        logger.info(f"✓ Loaded {len(df)} products from CSV")
        logger.info(f"  Columns: {list(df.columns)}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to load CSV: {e}")
        return False

def test_data_preparation():
    """Test data preparation."""
    logger.info("Testing data preparation...")
    
    try:
        from src.pipeline.ingestion import load_products_csv, prepare_product_data
        df = load_products_csv("data/products_catalog.csv")
        documents, metadatas, ids = prepare_product_data(df)
        
        logger.info(f"✓ Prepared {len(documents)} documents")
        logger.info(f"  Sample document: {documents[0][:100]}...")
        logger.info(f"  Sample metadata: {metadatas[0]}")
        logger.info(f"  Sample ID: {ids[0]}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to prepare data: {e}")
        return False

def test_vector_store():
    """Test vector store initialization."""
    logger.info("Testing vector store...")
    
    try:
        from src.pipeline.vector_store import get_vector_store
        vector_store = get_vector_store(
            persist_directory="./test_chroma_db",
            collection_name="test_products"
        )
        vector_store.get_or_create_collection()
        stats = vector_store.get_collection_stats()
        logger.info(f"✓ Vector store initialized: {stats}")
        
        # Clean up test collection
        vector_store.delete_collection()
        logger.info("✓ Test collection cleaned up")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to initialize vector store: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Starting ingestion pipeline tests")
    logger.info("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("CSV Loading", test_csv_loading),
        ("Data Preparation", test_data_preparation),
        ("Vector Store", test_vector_store),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} raised exception: {e}")
            results.append((test_name, False))
    
    logger.info("\n" + "=" * 60)
    logger.info("Test Results Summary")
    logger.info("=" * 60)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


