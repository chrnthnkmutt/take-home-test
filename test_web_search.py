"""
Test script for WebSearchTool implementation.

This script tests the web search tool with various queries to ensure
it works correctly with both mock and API implementations.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.tools.web_search import WebSearchTool


def test_basic_search():
    """Test basic search functionality."""
    print("=" * 80)
    print("TEST 1: Basic Search - Noise-Cancelling Headphones")
    print("=" * 80)
    
    tool = WebSearchTool()
    
    query = "What is the current market price for noise-cancelling headphones?"
    print(f"\nQuery: {query}")
    print(f"API Available: {tool.api_available}")
    print(f"LLM Available: {tool.llm_available}")
    
    try:
        results = tool.search(query)
        
        print(f"\n✓ Search completed successfully")
        print(f"  - API used: {results['metadata']['api_used']}")
        print(f"  - Results returned: {results['metadata']['n_results_returned']}")
        print(f"  - Timestamp: {results['metadata']['timestamp']}")
        
        print(f"\nAnswer Summary:")
        print("-" * 80)
        print(results['answer'])
        
        print(f"\nSearch Results:")
        print("-" * 80)
        for i, result in enumerate(results['results'], 1):
            print(f"\n{i}. {result['title']}")
            print(f"   URL: {result['url']}")
            print(f"   Relevance: {result['relevance_score']:.2f}")
            print(f"   Snippet: {result['snippet'][:150]}...")
        
        print(f"\nSources ({len(results['sources'])}):")
        for url in results['sources'][:3]:
            print(f"  - {url}")
        
        return True
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_product_reviews():
    """Test search for product reviews."""
    print("\n" + "=" * 80)
    print("TEST 2: Product Reviews - Sony WH-1000XM5")
    print("=" * 80)
    
    tool = WebSearchTool()
    
    query = "Latest reviews for Sony WH-1000XM5"
    print(f"\nQuery: {query}")
    
    try:
        results = tool.search(query, n_results=3)
        
        print(f"\n✓ Search completed successfully")
        print(f"  - Results returned: {results['metadata']['n_results_returned']}")
        
        print(f"\nAnswer Summary:")
        print("-" * 80)
        print(results['answer'])
        
        print(f"\nTop 3 Results:")
        for i, result in enumerate(results['results'], 1):
            print(f"\n{i}. {result['title']}")
            print(f"   Relevance: {result['relevance_score']:.2f}")
        
        return True
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        return False


def test_trending_products():
    """Test search for trending products."""
    print("\n" + "=" * 80)
    print("TEST 3: Trending Products - Home Fitness Equipment")
    print("=" * 80)
    
    tool = WebSearchTool()
    
    query = "Trending products in home fitness equipment"
    print(f"\nQuery: {query}")
    
    try:
        results = tool.search(query, n_results=5)
        
        print(f"\n✓ Search completed successfully")
        print(f"  - Results returned: {results['metadata']['n_results_returned']}")
        
        print(f"\nAnswer Summary:")
        print("-" * 80)
        print(results['answer'])
        
        return True
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        return False


def test_error_handling():
    """Test error handling."""
    print("\n" + "=" * 80)
    print("TEST 4: Error Handling - Empty Query")
    print("=" * 80)
    
    tool = WebSearchTool()
    
    try:
        results = tool.search("")
        print(f"\n✗ Test failed: Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"\n✓ Correctly raised ValueError: {e}")
        return True
    except Exception as e:
        print(f"\n✗ Test failed with unexpected error: {e}")
        return False


def test_stats():
    """Test get_stats method."""
    print("\n" + "=" * 80)
    print("TEST 5: Get Stats")
    print("=" * 80)
    
    tool = WebSearchTool()
    
    try:
        stats = tool.get_stats()
        
        print(f"\n✓ Stats retrieved successfully:")
        print(f"  - API Available: {stats['api_available']}")
        print(f"  - API Type: {stats['api_type']}")
        print(f"  - LLM Available: {stats['llm_available']}")
        print(f"  - Default N Results: {stats['default_n_results']}")
        
        return True
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("WEB SEARCH TOOL TEST SUITE")
    print("=" * 80)
    
    # Check environment
    print("\nEnvironment Check:")
    print(f"  - TAVILY_API_KEY: {'Set' if os.getenv('TAVILY_API_KEY') else 'Not set (will use mock)'}")
    print(f"  - GOOGLE_API_KEY: {'Set' if os.getenv('GOOGLE_API_KEY') else 'Not set (will use fallback)'}")
    
    # Run tests
    tests = [
        ("Basic Search", test_basic_search),
        ("Product Reviews", test_product_reviews),
        ("Trending Products", test_trending_products),
        ("Error Handling", test_error_handling),
        ("Get Stats", test_stats)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())


