"""
Example script demonstrating the ProductResearchAgent usage.

This script shows how to use the AI Agent to process various types of queries
and see how it intelligently routes to the appropriate tools.
"""

import sys
import os
import logging
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent.agent import ProductResearchAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def print_result(result: dict):
    """Pretty print agent result."""
    print("\n" + "="*80)
    print(f"QUERY: {result['query']}")
    print("="*80)
    print(f"\nREASONING: {result['reasoning']}")
    print(f"\nTOOLS USED: {', '.join(result['tools_used'])}")
    print(f"\nFINAL ANSWER:\n{result['final_answer']}")
    print(f"\nMETADATA:")
    print(f"  - Execution Time: {result['metadata']['execution_time_ms']:.2f}ms")
    print(f"  - Success: {result['metadata']['success']}")
    print("="*80 + "\n")


def main():
    """Run example queries through the agent."""
    # Load environment variables
    load_dotenv()
    
    print("\n🤖 Initializing ProductResearchAgent...")
    agent = ProductResearchAgent()
    
    # Get agent stats
    stats = agent.get_stats()
    print(f"\nAgent Configuration:")
    print(f"  - Routing Mode: {stats['agent']['routing_mode']}")
    print(f"  - LLM Available: {stats['agent']['llm_available']}")
    print(f"  - Catalog Tool Ready: {stats['tools']['catalog']['llm_available']}")
    print(f"  - Web Search API: {stats['tools']['web_search']['api_type']}")
    
    # Example queries demonstrating different routing scenarios
    example_queries = [
        # Single tool - Catalog
        "What wireless headphones do we have in stock?",
        
        # Single tool - Web Search
        "What's the current market price for Sony WH-1000XM5 headphones?",
        
        # Single tool - Price Analysis
        "Which products have profit margins below 40%?",
        
        # Multiple tools - Comprehensive analysis
        "Should we lower the price of our AudioMax Pro headphones based on competitor pricing?",
        
        # Another catalog query
        "Show me all fitness equipment under $500",
        
        # Price analysis query
        "Calculate the average profit margin for Electronics category"
    ]
    
    print("\n" + "="*80)
    print("RUNNING EXAMPLE QUERIES")
    print("="*80)
    
    for i, query in enumerate(example_queries, 1):
        print(f"\n\n📝 Example {i}/{len(example_queries)}")
        
        try:
            result = agent.process_query(query)
            print_result(result)
            
            # Show detailed results for multi-tool queries
            if len(result['tools_used']) > 1:
                print("DETAILED RESULTS FROM EACH TOOL:")
                for tool_name, tool_result in result['results'].items():
                    print(f"\n  {tool_name}:")
                    if 'error' in tool_result:
                        print(f"    Error: {tool_result['error']}")
                    else:
                        # Show key metrics from each tool
                        if tool_name == "ProductCatalogRAG":
                            products = tool_result.get('products', [])
                            print(f"    - Found {len(products)} products")
                            print(f"    - Confidence: {tool_result.get('confidence', 0):.2f}")
                        elif tool_name == "WebSearchTool":
                            results = tool_result.get('results', [])
                            print(f"    - Found {len(results)} web results")
                        elif tool_name == "PriceAnalysisTool":
                            calcs = tool_result.get('calculations', {})
                            print(f"    - Products analyzed: {calcs.get('products_with_cost_data', 0)}")
                            print(f"    - Average margin: {calcs.get('average_margin', 'N/A')}%")
                print()
        
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"❌ Error: {e}\n")
    
    print("\n✅ All examples completed!")


if __name__ == "__main__":
    main()


