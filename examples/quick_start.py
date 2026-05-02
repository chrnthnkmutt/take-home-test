"""
Quick Start Guide for AI Product Research Assistant

This script demonstrates the complete workflow:
1. Run data ingestion pipeline
2. Initialize the agent
3. Process various types of queries
4. Save results to database
5. Retrieve query history
6. Submit feedback

Run this script to see the entire system in action!
"""

import sys
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.ingestion import ingest_products
from src.agent.agent import ProductResearchAgent
from src.database import init_db
from src.database_operations import save_query, get_all_queries, save_feedback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def print_step(step_num: int, description: str):
    """Print a step description."""
    print(f"\n{'─'*80}")
    print(f"STEP {step_num}: {description}")
    print(f"{'─'*80}\n")


def print_result(label: str, value: str, indent: int = 0):
    """Print a labeled result."""
    prefix = "  " * indent
    print(f"{prefix}{label}: {value}")


def main():
    """Run the complete quick start workflow."""
    
    print_section("🚀 AI Product Research Assistant - Quick Start Guide")
    
    # Load environment variables
    load_dotenv()
    
    # Check for required API keys
    google_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    print("Environment Check:")
    print_result("Google API Key", "✓ Set" if google_key else "✗ Not set")
    print_result("OpenAI API Key", "✓ Set" if openai_key else "✗ Not set (will use fallback)")
    
    if not google_key:
        logger.warning("Google API Key not set. Agent will use rule-based routing.")
    
    # -------------------------------------------------------------------------
    # STEP 1: Run Data Ingestion Pipeline
    # -------------------------------------------------------------------------
    print_step(1, "Running Data Ingestion Pipeline")
    
    print("Loading products from CSV and generating embeddings...")
    print("This may take a minute on first run...\n")
    
    try:
        stats = ingest_products(
            csv_path="data/products_catalog.csv",
            persist_directory="./chroma_db",
            collection_name="products",
            use_openai=True,
            batch_size=50
        )
        
        print("✓ Ingestion completed successfully!")
        print_result("Products ingested", str(stats['ingested_products']))
        print_result("Embedding model", stats['embedding_model'])
        print_result("Total products", str(stats['total_products']))
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        print(f"✗ Ingestion failed: {e}")
        print("\nNote: If vector database already exists, this is expected.")
        print("Continuing with existing data...\n")
    
    # -------------------------------------------------------------------------
    # STEP 2: Initialize Database
    # -------------------------------------------------------------------------
    print_step(2, "Initializing Query Database")
    
    try:
        init_db()
        print("✓ Database initialized successfully!")
        print_result("Database location", "./queries.db")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        print(f"✗ Database initialization failed: {e}")
        return
    
    # -------------------------------------------------------------------------
    # STEP 3: Initialize Agent
    # -------------------------------------------------------------------------
    print_step(3, "Initializing AI Agent")
    
    try:
        agent = ProductResearchAgent()
        print("✓ Agent initialized successfully!")
        
        # Get agent stats
        stats = agent.get_stats()
        print("\nAgent Configuration:")
        print_result("Routing mode", stats['agent']['routing_mode'], indent=1)
        print_result("LLM available", str(stats['agent']['llm_available']), indent=1)
        print_result("Model", stats['agent']['model_name'], indent=1)
        
        print("\nTools Status:")
        print_result("Catalog Tool", "✓ Ready" if stats['tools']['catalog']['llm_available'] else "⚠ Degraded", indent=1)
        print_result("Web Search", f"✓ {stats['tools']['web_search']['api_type']}", indent=1)
        print_result("Price Analysis", "✓ Ready", indent=1)
        
    except Exception as e:
        logger.error(f"Agent initialization failed: {e}")
        print(f"✗ Agent initialization failed: {e}")
        return
    
    # -------------------------------------------------------------------------
    # STEP 4: Process Example Queries
    # -------------------------------------------------------------------------
    print_step(4, "Processing Example Queries")
    
    # Define example queries
    example_queries = [
        {
            "query": "What wireless headphones do we have under $200?",
            "description": "Product catalog search",
            "expected_tools": ["ProductCatalogRAG"]
        },
        {
            "query": "Which products have profit margins below 40%?",
            "description": "Price analysis",
            "expected_tools": ["PriceAnalysisTool"]
        },
        {
            "query": "What's the market price for Sony WH-1000XM5?",
            "description": "Web search for market research",
            "expected_tools": ["WebSearchTool"]
        },
        {
            "query": "Should we adjust pricing for AudioMax Pro headphones?",
            "description": "Comprehensive analysis (multi-tool)",
            "expected_tools": ["ProductCatalogRAG", "WebSearchTool", "PriceAnalysisTool"]
        }
    ]
    
    query_ids = []
    
    for i, example in enumerate(example_queries, 1):
        print(f"\n{'─'*80}")
        print(f"Query {i}/{len(example_queries)}: {example['description']}")
        print(f"{'─'*80}")
        print(f"\nQuestion: \"{example['query']}\"")
        
        try:
            # Process query
            result = agent.process_query(example['query'])
            
            if result['metadata']['success']:
                print("\n✓ Query processed successfully!")
                print_result("Tools used", ", ".join(result['tools_used']))
                print_result("Execution time", f"{result['metadata']['execution_time_ms']:.2f}ms")
                print_result("Reasoning", result['reasoning'][:100] + "..." if len(result['reasoning']) > 100 else result['reasoning'])
                
                # Show answer preview
                answer_preview = result['final_answer'][:200] + "..." if len(result['final_answer']) > 200 else result['final_answer']
                print(f"\nAnswer Preview:\n{answer_preview}")
                
                # Save to database
                query_id = save_query(
                    query_text=example['query'],
                    tools_used=result['tools_used'],
                    result=result,
                    response_time_ms=result['metadata']['execution_time_ms']
                )
                
                if query_id:
                    query_ids.append(query_id)
                    print_result("Query ID", query_id)
                
            else:
                print(f"\n✗ Query failed: {result['metadata'].get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"\n✗ Error: {e}")
    
    # -------------------------------------------------------------------------
    # STEP 5: Retrieve Query History
    # -------------------------------------------------------------------------
    print_step(5, "Retrieving Query History")
    
    try:
        all_queries = get_all_queries()
        print(f"✓ Retrieved {len(all_queries)} queries from database")
        
        if all_queries:
            print("\nRecent Queries:")
            for i, q in enumerate(all_queries[-5:], 1):  # Show last 5
                print(f"\n  {i}. Query ID: {q['id']}")
                print(f"     Text: {q['query_text'][:60]}...")
                print(f"     Tools: {', '.join(q['tools_used'])}")
                print(f"     Time: {q['timestamp']}")
                print(f"     Response Time: {q['response_time_ms']:.2f}ms")
                print(f"     Feedbacks: {len(q['feedbacks'])}")
        
    except Exception as e:
        logger.error(f"Error retrieving queries: {e}")
        print(f"✗ Error: {e}")
    
    # -------------------------------------------------------------------------
    # STEP 6: Submit Feedback
    # -------------------------------------------------------------------------
    print_step(6, "Submitting User Feedback")
    
    if query_ids:
        # Submit feedback for first query
        try:
            feedback_id = save_feedback(
                query_id=query_ids[0],
                rating=5,
                comment="Excellent! Found exactly what I needed."
            )
            
            if feedback_id:
                print("✓ Positive feedback submitted successfully!")
                print_result("Feedback ID", feedback_id)
                print_result("Query ID", query_ids[0])
                print_result("Rating", "5/5")
                print_result("Comment", "Excellent! Found exactly what I needed.")
            
        except Exception as e:
            logger.error(f"Error submitting feedback: {e}")
            print(f"✗ Error: {e}")
        
        # Submit feedback for second query if available
        if len(query_ids) > 1:
            try:
                feedback_id = save_feedback(
                    query_id=query_ids[1],
                    rating=4,
                    comment="Good results, but could be more detailed."
                )
                
                if feedback_id:
                    print("\n✓ Another feedback submitted!")
                    print_result("Feedback ID", feedback_id)
                    print_result("Query ID", query_ids[1])
                    print_result("Rating", "4/5")
                
            except Exception as e:
                logger.error(f"Error submitting feedback: {e}")
    else:
        print("⚠ No query IDs available for feedback submission")
    
    # -------------------------------------------------------------------------
    # STEP 7: Summary and Next Steps
    # -------------------------------------------------------------------------
    print_step(7, "Summary and Next Steps")
    
    print("✓ Quick start completed successfully!\n")
    
    print("What you've accomplished:")
    print("  1. ✓ Loaded product data and generated embeddings")
    print("  2. ✓ Initialized the query database")
    print("  3. ✓ Created an AI agent with intelligent routing")
    print("  4. ✓ Processed multiple types of queries")
    print("  5. ✓ Saved queries to database")
    print("  6. ✓ Retrieved query history")
    print("  7. ✓ Submitted user feedback")
    
    print("\n" + "─"*80)
    print("Next Steps:")
    print("─"*80)
    
    print("\n1. Start the API server:")
    print("   uvicorn src.api.main:app --reload")
    
    print("\n2. Test the API:")
    print("   ./examples/test_api.sh")
    print("   (or manually with curl/Postman)")
    
    print("\n3. Explore API documentation:")
    print("   http://localhost:8000/docs")
    
    print("\n4. Try more queries:")
    print("   python examples/run_agent.py")
    
    print("\n5. Run tests:")
    print("   pytest tests/ -v")
    
    print("\n" + "─"*80)
    print("Useful Resources:")
    print("─"*80)
    
    print("\n  • README.md - Complete project documentation")
    print("  • AGENT.md - Agent routing and tool selection")
    print("  • ARCHITECTURE.md - System architecture details")
    print("  • API Docs - http://localhost:8000/docs (when server is running)")
    
    print("\n" + "="*80)
    print("  🎉 You're all set! Happy researching!")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)


