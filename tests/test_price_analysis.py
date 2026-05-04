"""
Test script for the Price Analysis tool.

This script demonstrates the functionality of the PriceAnalysisTool
including deterministic calculations and LLM-based insights.
"""

import logging
from src.tools.price_analysis import (
    PriceAnalysisTool,
    calculate_margin,
    calculate_markup,
    calculate_profit
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_calculation_functions():
    """Test the deterministic calculation functions."""
    print("\n" + "="*80)
    print("Testing Deterministic Calculation Functions")
    print("="*80)
    
    # Test case 1: Normal case
    price = 100.0
    cost = 60.0
    
    margin = calculate_margin(price, cost)
    markup = calculate_markup(price, cost)
    profit = calculate_profit(price, cost)
    
    print(f"\nTest Case 1: Price=${price}, Cost=${cost}")
    print(f"  Margin: {margin}% (Expected: 40.0%)")
    print(f"  Markup: {markup}% (Expected: 66.67%)")
    print(f"  Profit: ${profit} (Expected: $40.0)")
    
    assert margin == 40.0, f"Margin calculation failed: {margin} != 40.0"
    assert markup == 66.67, f"Markup calculation failed: {markup} != 66.67"
    assert profit == 40.0, f"Profit calculation failed: {profit} != 40.0"
    
    # Test case 2: High margin
    price = 200.0
    cost = 50.0
    
    margin = calculate_margin(price, cost)
    markup = calculate_markup(price, cost)
    profit = calculate_profit(price, cost)
    
    print(f"\nTest Case 2: Price=${price}, Cost=${cost}")
    print(f"  Margin: {margin}% (Expected: 75.0%)")
    print(f"  Markup: {markup}% (Expected: 300.0%)")
    print(f"  Profit: ${profit} (Expected: $150.0)")
    
    assert margin == 75.0, f"Margin calculation failed: {margin} != 75.0"
    assert markup == 300.0, f"Markup calculation failed: {markup} != 300.0"
    assert profit == 150.0, f"Profit calculation failed: {profit} != 150.0"
    
    # Test case 3: Low margin
    price = 100.0
    cost = 95.0
    
    margin = calculate_margin(price, cost)
    markup = calculate_markup(price, cost)
    profit = calculate_profit(price, cost)
    
    print(f"\nTest Case 3: Price=${price}, Cost=${cost}")
    print(f"  Margin: {margin}% (Expected: 5.0%)")
    print(f"  Markup: {markup}% (Expected: 5.26%)")
    print(f"  Profit: ${profit} (Expected: $5.0)")
    
    assert margin == 5.0, f"Margin calculation failed: {margin} != 5.0"
    assert markup == 5.26, f"Markup calculation failed: {markup} != 5.26"
    assert profit == 5.0, f"Profit calculation failed: {profit} != 5.0"
    
    # Test case 4: Error handling - zero price
    print(f"\nTest Case 4: Error handling - zero price")
    try:
        calculate_margin(0, 50)
        print("  ERROR: Should have raised ValueError")
        assert False, "Should have raised ValueError for zero price"
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")
    
    # Test case 5: Error handling - zero cost
    print(f"\nTest Case 5: Error handling - zero cost")
    try:
        calculate_markup(100, 0)
        print("  ERROR: Should have raised ValueError")
        assert False, "Should have raised ValueError for zero cost"
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")
    
    # Test case 6: Error handling - negative values
    print(f"\nTest Case 6: Error handling - negative cost")
    try:
        calculate_profit(100, -10)
        print("  ERROR: Should have raised ValueError")
        assert False, "Should have raised ValueError for negative cost"
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")
    
    print("\n✓ All calculation function tests passed!")


def test_price_analysis_tool():
    """Test the PriceAnalysisTool with vector store."""
    print("\n" + "="*80)
    print("Testing PriceAnalysisTool")
    print("="*80)
    
    try:
        # Initialize the tool
        print("\nInitializing PriceAnalysisTool...")
        tool = PriceAnalysisTool(
            persist_directory="./chroma_db",
            collection_name="products"
        )
        
        # Check if vector store is initialized
        if not tool.vector_store.is_initialized():
            print("\n⚠ Warning: Vector store is not initialized with data.")
            print("Please run the ingestion pipeline first:")
            print("  python examples/run_ingestion.py")
            return
        
        # Get collection stats
        stats = tool.vector_store.get_collection_stats()
        print(f"\nVector Store Stats:")
        print(f"  Collection: {stats['collection_name']}")
        print(f"  Documents: {stats['document_count']}")
        
        # Test query 1: General margin analysis
        print("\n" + "-"*80)
        print("Test Query 1: Which products have the lowest profit margins?")
        print("-"*80)
        
        result = tool.analyze(
            query="Which products have the lowest profit margins?",
            n_results=10
        )
        
        print(f"\nAnalysis:")
        print(result['analysis'])
        
        print(f"\nCalculations:")
        for key, value in result['calculations'].items():
            print(f"  {key}: {value}")
        
        print(f"\nTop 3 Products:")
        for i, product in enumerate(result['products'][:3], 1):
            print(f"  {i}. {product['name']}")
            print(f"     Price: ${product['price']}, Cost: ${product.get('cost', 'N/A')}")
            if product.get('margin') is not None:
                print(f"     Margin: {product['margin']}%, Markup: {product['markup']}%, Profit: ${product['profit']}")
            else:
                print(f"     {product.get('calculation_error', 'No cost data')}")
        
        # Test query 2: Category-specific analysis
        print("\n" + "-"*80)
        print("Test Query 2: Calculate average margin for Electronics category")
        print("-"*80)
        
        result = tool.analyze(
            query="Calculate average margin for Electronics category",
            category="Electronics",
            n_results=20
        )
        
        print(f"\nAnalysis:")
        print(result['analysis'])
        
        print(f"\nCalculations:")
        for key, value in result['calculations'].items():
            print(f"  {key}: {value}")
        
        # Test query 3: Margin threshold filter
        print("\n" + "-"*80)
        print("Test Query 3: Show me products with margins below 40%")
        print("-"*80)
        
        result = tool.analyze(
            query="Show me products with margins below 40%",
            margin_threshold=40.0,
            n_results=20
        )
        
        print(f"\nAnalysis:")
        print(result['analysis'])
        
        print(f"\nCalculations:")
        for key, value in result['calculations'].items():
            print(f"  {key}: {value}")
        
        print(f"\nProducts with margin < 40%:")
        for i, product in enumerate(result['products'][:5], 1):
            if product.get('margin') is not None:
                print(f"  {i}. {product['name']}: {product['margin']}% margin")
        
        print("\n✓ PriceAnalysisTool tests completed!")
        
    except Exception as e:
        logger.error(f"Error during PriceAnalysisTool test: {e}", exc_info=True)
        print(f"\n✗ Test failed with error: {e}")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("Price Analysis Tool - Test Suite")
    print("="*80)
    
    # Test calculation functions
    test_calculation_functions()
    
    # Test the full tool
    test_price_analysis_tool()
    
    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)


if __name__ == "__main__":
    main()


