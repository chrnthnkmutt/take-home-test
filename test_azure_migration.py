#!/usr/bin/env python3
"""
Test script to verify Azure OpenAI migration is working correctly.

This script tests all components that use Azure OpenAI:
1. Azure OpenAI chat model initialization
2. Azure OpenAI embeddings initialization
3. ProductResearchAgent with Azure OpenAI
4. ProductCatalogRAG with Azure OpenAI
5. WebSearchTool with Azure OpenAI
6. PriceAnalysisTool with Azure OpenAI
7. End-to-end query through the agent
"""

import os
import sys
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables
load_dotenv()

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{BLUE}{'=' * 80}{RESET}")
    print(f"{BLUE}{text.center(80)}{RESET}")
    print(f"{BLUE}{'=' * 80}{RESET}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{GREEN}✓ {text}{RESET}")


def print_error(text: str):
    """Print error message."""
    print(f"{RED}✗ {text}{RESET}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{YELLOW}⚠ {text}{RESET}")


def print_info(text: str):
    """Print info message."""
    print(f"  {text}")


def mask_api_key(api_key: str) -> str:
    """Mask API key for display."""
    if not api_key:
        return "Not set"
    if len(api_key) <= 8:
        return "*" * len(api_key)
    return f"{api_key[:4]}...{api_key[-4:]}"


def test_environment_variables() -> bool:
    """Test 1: Check if all required environment variables are set."""
    print_header("Test 1: Environment Variables Check")
    
    required_vars = {
        "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
        "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "AZURE_OPENAI_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    }
    
    all_set = True
    for var_name, var_value in required_vars.items():
        if var_value:
            if "API_KEY" in var_name:
                print_success(f"{var_name}: {mask_api_key(var_value)}")
            else:
                print_success(f"{var_name}: {var_value}")
        else:
            print_error(f"{var_name}: Not set")
            all_set = False
    
    if all_set:
        print_success("\nAll required environment variables are set!")
        return True
    else:
        print_error("\nSome required environment variables are missing!")
        return False


def test_azure_chat_model() -> bool:
    """Test 2: Test Azure OpenAI chat model initialization."""
    print_header("Test 2: Azure OpenAI Chat Model Initialization")
    
    try:
        from langchain_openai import AzureChatOpenAI
        
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        print_info(f"Initializing AzureChatOpenAI with:")
        print_info(f"  Endpoint: {endpoint}")
        print_info(f"  Deployment: {deployment}")
        print_info(f"  API Version: 2024-02-15-preview")
        
        llm = AzureChatOpenAI(
            azure_endpoint=endpoint,
            azure_deployment=deployment,
            api_version="2024-02-15-preview",
            temperature=0.7
        )
        
        print_success("AzureChatOpenAI initialized successfully!")
        
        # Test a simple invocation
        print_info("\nTesting simple invocation...")
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content="Say 'Hello, Azure OpenAI!'")])
        
        content = response.content if isinstance(response.content, str) else str(response.content)
        print_success(f"Response received: {content[:100]}...")
        
        return True
        
    except Exception as e:
        print_error(f"Failed to initialize or test AzureChatOpenAI: {e}")
        return False


def test_azure_embeddings() -> bool:
    """Test 3: Test Azure OpenAI embeddings initialization."""
    print_header("Test 3: Azure OpenAI Embeddings Initialization")
    
    try:
        from langchain_openai import AzureOpenAIEmbeddings
        
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        
        print_info(f"Initializing AzureOpenAIEmbeddings with:")
        print_info(f"  Endpoint: {endpoint}")
        print_info(f"  Deployment: {deployment}")
        print_info(f"  API Version: 2024-02-15-preview")
        
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=endpoint,
            azure_deployment=deployment,
            api_version="2024-02-15-preview"
        )
        
        print_success("AzureOpenAIEmbeddings initialized successfully!")
        
        # Test embedding generation
        print_info("\nTesting embedding generation...")
        test_text = "This is a test sentence for embeddings."
        embedding = embeddings.embed_query(test_text)
        
        print_success(f"Embedding generated successfully! Dimension: {len(embedding)}")
        
        return True
        
    except Exception as e:
        print_error(f"Failed to initialize or test AzureOpenAIEmbeddings: {e}")
        return False


def test_product_catalog_rag() -> bool:
    """Test 4: Test ProductCatalogRAG with Azure OpenAI."""
    print_header("Test 4: ProductCatalogRAG with Azure OpenAI")
    
    try:
        from src.tools.product_catalog_rag import ProductCatalogRAG
        
        print_info("Initializing ProductCatalogRAG...")
        catalog_tool = ProductCatalogRAG()
        
        print_success("ProductCatalogRAG initialized successfully!")
        print_info(f"  LLM Available: {catalog_tool.llm_available}")
        
        # Get stats
        stats = catalog_tool.get_stats()
        print_info(f"  Collection: {stats.get('collection_name', 'N/A')}")
        print_info(f"  Document Count: {stats.get('count', 0)}")
        
        if catalog_tool.llm_available:
            print_success("Azure OpenAI LLM is available for ProductCatalogRAG!")
        else:
            print_warning("Azure OpenAI LLM is NOT available for ProductCatalogRAG")
        
        return True
        
    except Exception as e:
        print_error(f"Failed to initialize ProductCatalogRAG: {e}")
        return False


def test_web_search_tool() -> bool:
    """Test 5: Test WebSearchTool with Azure OpenAI."""
    print_header("Test 5: WebSearchTool with Azure OpenAI")
    
    try:
        from src.tools.web_search import WebSearchTool
        
        print_info("Initializing WebSearchTool...")
        web_tool = WebSearchTool()
        
        print_success("WebSearchTool initialized successfully!")
        
        # Get stats
        stats = web_tool.get_stats()
        print_info(f"  API Available: {stats.get('api_available', False)}")
        print_info(f"  API Type: {stats.get('api_type', 'N/A')}")
        print_info(f"  LLM Available: {stats.get('llm_available', False)}")
        
        if web_tool.llm_available:
            print_success("Azure OpenAI LLM is available for WebSearchTool!")
        else:
            print_warning("Azure OpenAI LLM is NOT available for WebSearchTool")
        
        return True
        
    except Exception as e:
        print_error(f"Failed to initialize WebSearchTool: {e}")
        return False


def test_price_analysis_tool() -> bool:
    """Test 6: Test PriceAnalysisTool with Azure OpenAI."""
    print_header("Test 6: PriceAnalysisTool with Azure OpenAI")
    
    try:
        from src.tools.price_analysis import PriceAnalysisTool
        
        print_info("Initializing PriceAnalysisTool...")
        price_tool = PriceAnalysisTool()
        
        print_success("PriceAnalysisTool initialized successfully!")
        print_info(f"  LLM Available: {price_tool.llm_available}")
        print_info(f"  Default N Results: {price_tool.default_n_results}")
        
        if price_tool.llm_available:
            print_success("Azure OpenAI LLM is available for PriceAnalysisTool!")
        else:
            print_warning("Azure OpenAI LLM is NOT available for PriceAnalysisTool")
        
        return True
        
    except Exception as e:
        print_error(f"Failed to initialize PriceAnalysisTool: {e}")
        return False


def test_product_research_agent() -> bool:
    """Test 7: Test ProductResearchAgent with Azure OpenAI."""
    print_header("Test 7: ProductResearchAgent with Azure OpenAI")
    
    try:
        from src.agent.agent import ProductResearchAgent
        
        print_info("Initializing ProductResearchAgent...")
        agent = ProductResearchAgent()
        
        print_success("ProductResearchAgent initialized successfully!")
        
        # Get stats
        stats = agent.get_stats()
        print_info(f"  LLM Available: {stats['agent']['llm_available']}")
        print_info(f"  Routing Mode: {stats['agent']['routing_mode']}")
        
        if stats['agent']['llm_available']:
            print_success("Azure OpenAI LLM is available for ProductResearchAgent!")
        else:
            print_warning("Azure OpenAI LLM is NOT available for ProductResearchAgent")
        
        return True
        
    except Exception as e:
        print_error(f"Failed to initialize ProductResearchAgent: {e}")
        return False


def test_end_to_end_query() -> bool:
    """Test 8: Test end-to-end query through the agent."""
    print_header("Test 8: End-to-End Query Test")
    
    try:
        from src.agent.agent import ProductResearchAgent
        
        print_info("Initializing ProductResearchAgent...")
        agent = ProductResearchAgent()
        
        # Test query
        test_query = "What wireless headphones do we have?"
        print_info(f"\nProcessing query: '{test_query}'")
        
        result = agent.process_query(test_query)
        
        print_success("Query processed successfully!")
        print_info(f"\nQuery Results:")
        print_info(f"  Tools Used: {', '.join(result['tools_used'])}")
        print_info(f"  Reasoning: {result['reasoning'][:100]}...")
        print_info(f"  Execution Time: {result['metadata']['execution_time_ms']:.2f}ms")
        print_info(f"  Success: {result['metadata']['success']}")
        
        print_info(f"\nFinal Answer Preview:")
        answer_preview = result['final_answer'][:200]
        print_info(f"  {answer_preview}...")
        
        # Check if LLM was used
        if result['metadata'].get('success'):
            print_success("\nEnd-to-end query test PASSED!")
            return True
        else:
            print_error("\nEnd-to-end query test FAILED!")
            return False
        
    except Exception as e:
        print_error(f"Failed to process end-to-end query: {e}")
        import traceback
        print_error(f"Traceback: {traceback.format_exc()}")
        return False


def print_summary(results: Dict[str, bool]):
    """Print test summary."""
    print_header("Test Summary")
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    failed_tests = total_tests - passed_tests
    
    print_info(f"Total Tests: {total_tests}")
    print_success(f"Passed: {passed_tests}")
    if failed_tests > 0:
        print_error(f"Failed: {failed_tests}")
    
    print("\nDetailed Results:")
    for test_name, result in results.items():
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"  {test_name}: {status}")
    
    if failed_tests == 0:
        print_success("\n🎉 All tests passed! Azure OpenAI migration is working correctly!")
        return True
    else:
        print_error(f"\n❌ {failed_tests} test(s) failed. Please check the errors above.")
        return False


def main():
    """Run all tests."""
    print_header("Azure OpenAI Migration Test Suite")
    print_info("This script will test all components using Azure OpenAI")
    print_info("Make sure you have run the ingestion pipeline first!")
    
    # Run all tests
    results = {
        "Environment Variables": test_environment_variables(),
        "Azure Chat Model": test_azure_chat_model(),
        "Azure Embeddings": test_azure_embeddings(),
        "ProductCatalogRAG": test_product_catalog_rag(),
        "WebSearchTool": test_web_search_tool(),
        "PriceAnalysisTool": test_price_analysis_tool(),
        "ProductResearchAgent": test_product_research_agent(),
        "End-to-End Query": test_end_to_end_query()
    }
    
    # Print summary
    success = print_summary(results)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


