"""
AI Agent with routing logic for the AI Product Research Assistant.

This module provides an intelligent agent that coordinates all three tools
(ProductCatalogRAG, WebSearchTool, PriceAnalysisTool) and autonomously decides
which tool(s) to use based on query analysis.
"""

import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.tools.product_catalog_rag import ProductCatalogRAG
from src.tools.web_search import WebSearchTool
from src.tools.price_analysis import PriceAnalysisTool

logger = logging.getLogger(__name__)


class ProductResearchAgent:
    """
    AI Agent that coordinates multiple tools for product research.
    
    The agent analyzes user queries and autonomously decides which tool(s) to use,
    executes them in the appropriate order, and aggregates results into a
    comprehensive response.
    
    Attributes:
        catalog_tool: ProductCatalogRAG instance for catalog search
        web_tool: WebSearchTool instance for web search
        price_tool: PriceAnalysisTool instance for price analysis
        llm: LangChain AzureChatOpenAI instance for query analysis
        llm_available: Whether LLM is available for routing
    """
    
    def __init__(
        self,
        catalog_tool: Optional[ProductCatalogRAG] = None,
        web_tool: Optional[WebSearchTool] = None,
        price_tool: Optional[PriceAnalysisTool] = None,
        azure_api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_deployment_name: Optional[str] = None,
        temperature: float = 0.3
    ):
        """
        Initialize the ProductResearchAgent.
        
        Args:
            catalog_tool: Optional pre-initialized ProductCatalogRAG instance
            web_tool: Optional pre-initialized WebSearchTool instance
            price_tool: Optional pre-initialized PriceAnalysisTool instance
            azure_api_key: Azure OpenAI API key (uses AZURE_OPENAI_API_KEY env var if not provided)
            azure_endpoint: Azure OpenAI endpoint (uses AZURE_OPENAI_ENDPOINT env var if not provided)
            azure_deployment_name: Azure OpenAI deployment name (uses AZURE_OPENAI_DEPLOYMENT_NAME env var if not provided)
            temperature: LLM temperature (lower for more deterministic routing)
        """
        # Initialize tools
        self.catalog_tool = catalog_tool or ProductCatalogRAG()
        self.web_tool = web_tool or WebSearchTool()
        self.price_tool = price_tool or PriceAnalysisTool()
        
        # Initialize LLM for query analysis and routing
        api_key = azure_api_key or os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment_name = azure_deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        self.llm = None
        self.llm_available = False
        
        if api_key and endpoint and deployment_name:
            try:
                # Set environment variable for API key if provided
                if api_key:
                    os.environ["AZURE_OPENAI_API_KEY"] = api_key
                
                self.llm = AzureChatOpenAI(
                    azure_endpoint=endpoint,
                    azure_deployment=deployment_name,
                    api_version="2024-02-15-preview",
                    temperature=temperature
                )
                self.llm_available = True
                logger.info(f"Agent LLM initialized with Azure OpenAI deployment: {deployment_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize agent LLM: {e}. Will use rule-based routing.")
                self.llm_available = False
        else:
            logger.warning("Azure OpenAI credentials not fully provided. Agent will use rule-based routing.")
        
        logger.info("ProductResearchAgent initialized successfully")
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query by analyzing intent and routing to appropriate tool(s).
        
        This method:
        1. Analyzes the user query to understand intent
        2. Decides which tool(s) to use and in what order
        3. Executes the selected tool(s)
        4. Aggregates results if multiple tools are used
        5. Returns a structured response with reasoning
        
        Args:
            query: Natural language user query
        
        Returns:
            Dictionary containing:
                - query: Original user query
                - reasoning: Explanation of tool selection
                - tools_used: List of tools that were used
                - results: Dictionary of results from each tool
                - final_answer: Aggregated answer combining all tool results
                - metadata: Execution metadata (timestamp, execution_time_ms, success)
        
        Raises:
            ValueError: If query is empty or invalid
            Exception: For other processing errors
        
        Examples:
            >>> agent = ProductResearchAgent()
            >>> result = agent.process_query("What wireless headphones do we have?")
            >>> print(result["reasoning"])
            >>> print(result["final_answer"])
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        start_time = time.time()
        
        try:
            # Step 1: Analyze query and select tools
            tool_selection = self._analyze_query_and_select_tools(query)
            
            logger.info(f"Query analysis complete. Tools selected: {tool_selection['tools']}")
            logger.info(f"Reasoning: {tool_selection['reasoning']}")
            
            # Step 2: Execute selected tools
            results = self._execute_tools(query, tool_selection)
            
            # Step 3: Aggregate results and generate final answer
            final_answer = self._aggregate_results(query, tool_selection, results)
            
            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Build response
            response = {
                "query": query,
                "reasoning": tool_selection["reasoning"],
                "tools_used": tool_selection["tools"],
                "results": results,
                "final_answer": final_answer,
                "metadata": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "execution_time_ms": round(execution_time_ms, 2),
                    "success": True
                }
            }
            
            logger.info(f"Query processed successfully in {execution_time_ms:.2f}ms")
            return response
        
        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            execution_time_ms = (time.time() - start_time) * 1000
            return self._create_error_response(query, str(e), execution_time_ms)
    
    def _analyze_query_and_select_tools(self, query: str) -> Dict[str, Any]:
        """
        Analyze query and determine which tool(s) to use.
        
        Uses LLM if available, otherwise falls back to rule-based selection.
        
        Args:
            query: User query
        
        Returns:
            Dictionary with:
                - tools: List of tool names to use (in order)
                - reasoning: Explanation of why these tools were selected
                - execution_order: Order of execution (sequential or parallel)
        """
        if self.llm_available and self.llm:
            try:
                return self._llm_based_tool_selection(query)
            except Exception as e:
                logger.warning(f"LLM-based tool selection failed: {e}. Using rule-based fallback.")
                return self._rule_based_tool_selection(query)
        else:
            return self._rule_based_tool_selection(query)
    
    def _llm_based_tool_selection(self, query: str) -> Dict[str, Any]:
        """
        Use LLM to analyze query and select appropriate tools.
        
        Args:
            query: User query
        
        Returns:
            Tool selection dictionary
        """
        system_message = SystemMessage(content="""You are an intelligent routing agent for a product research assistant.
Your task is to analyze user queries and determine which tool(s) to use.

Available tools:
1. ProductCatalogRAG - Search internal product catalog, get product details, specifications, availability
   Use for: "What products do we have?", "Show me headphones", "Product specifications", "Stock status"

2. WebSearchTool - Search the web for market information, competitor prices, reviews, trends
   Use for: "Market price for X", "Competitor analysis", "Product reviews", "Industry trends"

3. PriceAnalysisTool - Analyze pricing, margins, markups, profitability
   Use for: "Profit margins", "Which products have low margins?", "Price analysis", "Markup calculations"

You can select:
- Single tool: For straightforward queries
- Multiple tools (sequential): When query needs information from multiple sources
  Example: "Should we lower price?" → catalog (get our product) → web (competitor prices) → price (analyze margins)

Respond in this exact format:
TOOLS: [tool1, tool2, ...]
REASONING: Brief explanation of why these tools were selected and in this order
EXECUTION: sequential OR parallel""")
        
        human_message = HumanMessage(content=f"""Analyze this query and select the appropriate tool(s):

Query: "{query}"

Consider:
- What information does the user need?
- Which tool(s) can provide that information?
- Do multiple tools need to work together?
- What order should they execute in?""")
        
        response = self.llm.invoke([system_message, human_message])
        # Handle both string and list responses from Azure OpenAI
        content = response.content if isinstance(response.content, str) else str(response.content[0]) if response.content else ""
        content = content.strip()
        
        # Parse LLM response
        tools = []
        reasoning = ""
        execution_order = "sequential"
        
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("TOOLS:"):
                tools_str = line.replace("TOOLS:", "").strip()
                # Extract tool names from the string
                tools_str = tools_str.strip("[]")
                tools = [t.strip().strip("'\"") for t in tools_str.split(",") if t.strip()]
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()
            elif line.startswith("EXECUTION:"):
                execution_order = line.replace("EXECUTION:", "").strip().lower()
        
        # Validate tool names
        valid_tools = ["ProductCatalogRAG", "WebSearchTool", "PriceAnalysisTool"]
        tools = [t for t in tools if t in valid_tools]
        
        if not tools:
            logger.warning("LLM did not select valid tools. Using rule-based fallback.")
            return self._rule_based_tool_selection(query)
        
        return {
            "tools": tools,
            "reasoning": reasoning or "LLM-based tool selection",
            "execution_order": execution_order
        }
    
    def _rule_based_tool_selection(self, query: str) -> Dict[str, Any]:
        """
        Use rule-based logic to select tools based on keywords.
        
        Args:
            query: User query
        
        Returns:
            Tool selection dictionary
        """
        query_lower = query.lower()
        tools = []
        reasoning_parts = []
        
        # Keywords for each tool
        catalog_keywords = ["what", "show", "list", "have", "stock", "available", "product", "specification", "spec", "feature"]
        web_keywords = ["market", "competitor", "price for", "review", "trend", "industry", "compare", "vs", "versus"]
        price_keywords = ["margin", "markup", "profit", "cost", "pricing", "price analysis", "low margin", "high margin"]
        
        # Check for price analysis keywords
        if any(keyword in query_lower for keyword in price_keywords):
            tools.append("PriceAnalysisTool")
            reasoning_parts.append("Query involves pricing/margin analysis")
        
        # Check for web search keywords
        if any(keyword in query_lower for keyword in web_keywords):
            tools.append("WebSearchTool")
            reasoning_parts.append("Query requires market/competitor information")
        
        # Check for catalog keywords
        if any(keyword in query_lower for keyword in catalog_keywords):
            tools.append("ProductCatalogRAG")
            reasoning_parts.append("Query asks about internal product catalog")
        
        # Special case: pricing decision queries need all three tools
        decision_keywords = ["should we", "recommend", "adjust price", "change price", "lower price", "raise price"]
        if any(keyword in query_lower for keyword in decision_keywords):
            tools = ["ProductCatalogRAG", "WebSearchTool", "PriceAnalysisTool"]
            reasoning_parts = ["Query requires comprehensive analysis: catalog data, market research, and pricing analysis"]
        
        # Default to catalog if no tools selected
        if not tools:
            tools = ["ProductCatalogRAG"]
            reasoning_parts = ["Default to catalog search for general product queries"]
        
        reasoning = "; ".join(reasoning_parts)
        
        return {
            "tools": tools,
            "reasoning": reasoning,
            "execution_order": "sequential" if len(tools) > 1 else "single"
        }
    
    def _execute_tools(
        self,
        query: str,
        tool_selection: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute the selected tools in the appropriate order.
        
        Args:
            query: User query
            tool_selection: Tool selection dictionary
        
        Returns:
            Dictionary mapping tool names to their results
        """
        results = {}
        tools = tool_selection["tools"]
        
        for tool_name in tools:
            try:
                logger.info(f"Executing tool: {tool_name}")
                
                if tool_name == "ProductCatalogRAG":
                    result = self.catalog_tool.search(query)
                    results["ProductCatalogRAG"] = result
                
                elif tool_name == "WebSearchTool":
                    result = self.web_tool.search(query)
                    results["WebSearchTool"] = result
                
                elif tool_name == "PriceAnalysisTool":
                    result = self.price_tool.analyze(query)
                    results["PriceAnalysisTool"] = result
                
                logger.info(f"Tool {tool_name} executed successfully")
            
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}")
                # Continue with other tools, store error
                results[tool_name] = {
                    "error": str(e),
                    "success": False
                }
        
        return results
    
    def _aggregate_results(
        self,
        query: str,
        tool_selection: Dict[str, Any],
        results: Dict[str, Any]
    ) -> str:
        """
        Aggregate results from multiple tools into a final answer.
        
        Args:
            query: Original user query
            tool_selection: Tool selection information
            results: Results from each tool
        
        Returns:
            Aggregated final answer string
        """
        if not results:
            return "No results were obtained from the tools."
        
        # If only one tool was used, return its answer directly
        if len(results) == 1:
            tool_name = list(results.keys())[0]
            result = results[tool_name]
            
            if "error" in result:
                return f"Error from {tool_name}: {result['error']}"
            
            # Extract answer based on tool type
            if tool_name == "ProductCatalogRAG":
                return result.get("answer", "No answer available")
            elif tool_name == "WebSearchTool":
                return result.get("answer", "No answer available")
            elif tool_name == "PriceAnalysisTool":
                return result.get("analysis", "No analysis available")
        
        # Multiple tools - aggregate using LLM if available
        if self.llm_available and self.llm:
            try:
                return self._llm_aggregate_results(query, tool_selection, results)
            except Exception as e:
                logger.warning(f"LLM aggregation failed: {e}. Using simple aggregation.")
                return self._simple_aggregate_results(query, results)
        else:
            return self._simple_aggregate_results(query, results)
    
    def _llm_aggregate_results(
        self,
        query: str,
        tool_selection: Dict[str, Any],
        results: Dict[str, Any]
    ) -> str:
        """
        Use LLM to aggregate results from multiple tools.
        
        Args:
            query: Original query
            tool_selection: Tool selection info
            results: Results from tools
        
        Returns:
            Aggregated answer
        """
        # Build context from all tool results
        context_parts = []
        
        for tool_name, result in results.items():
            if "error" in result:
                context_parts.append(f"\n{tool_name} Error: {result['error']}")
                continue
            
            context_parts.append(f"\n=== {tool_name} Results ===")
            
            if tool_name == "ProductCatalogRAG":
                context_parts.append(f"Answer: {result.get('answer', 'N/A')}")
                products = result.get('products', [])
                if products:
                    context_parts.append(f"Found {len(products)} products")
                    for p in products[:3]:
                        context_parts.append(f"  - {p.get('product_name', 'Unknown')}: ${p.get('price', 0)}")
            
            elif tool_name == "WebSearchTool":
                context_parts.append(f"Answer: {result.get('answer', 'N/A')}")
                search_results = result.get('results', [])
                if search_results:
                    context_parts.append(f"Found {len(search_results)} web results")
            
            elif tool_name == "PriceAnalysisTool":
                context_parts.append(f"Analysis: {result.get('analysis', 'N/A')}")
                calcs = result.get('calculations', {})
                if calcs:
                    context_parts.append(f"Average Margin: {calcs.get('average_margin', 'N/A')}%")
                    context_parts.append(f"Total Profit: ${calcs.get('total_profit', 'N/A')}")
        
        context = "\n".join(context_parts)
        
        system_message = SystemMessage(content="""You are a product research assistant that synthesizes information from multiple tools.
Your task is to combine results from different sources into a comprehensive, actionable answer.
Be clear, concise, and provide specific recommendations when appropriate.""")
        
        human_message = HumanMessage(content=f"""User Query: "{query}"

Tool Selection Reasoning: {tool_selection['reasoning']}

Results from Tools:
{context}

Please provide a comprehensive answer that:
1. Directly addresses the user's query
2. Synthesizes information from all available sources
3. Provides clear recommendations if applicable
4. Highlights key insights and data points

Keep the response focused and actionable.""")
        
        response = self.llm.invoke([system_message, human_message])
        # Handle both string and list responses from Azure OpenAI
        content = response.content if isinstance(response.content, str) else str(response.content[0]) if response.content else ""
        return content.strip()
    
    def _simple_aggregate_results(
        self,
        query: str,
        results: Dict[str, Any]
    ) -> str:
        """
        Simple aggregation without LLM.
        
        Args:
            query: Original query
            results: Results from tools
        
        Returns:
            Simple aggregated answer
        """
        answer_parts = [f"Results for: {query}\n"]
        
        for tool_name, result in results.items():
            if "error" in result:
                answer_parts.append(f"\n{tool_name}: Error - {result['error']}")
                continue
            
            answer_parts.append(f"\n=== {tool_name} ===")
            
            if tool_name == "ProductCatalogRAG":
                answer_parts.append(result.get("answer", "No answer available"))
            elif tool_name == "WebSearchTool":
                answer_parts.append(result.get("answer", "No answer available"))
            elif tool_name == "PriceAnalysisTool":
                answer_parts.append(result.get("analysis", "No analysis available"))
        
        return "\n".join(answer_parts)
    
    def _create_error_response(
        self,
        query: str,
        error_message: str,
        execution_time_ms: float
    ) -> Dict[str, Any]:
        """
        Create error response structure.
        
        Args:
            query: Original query
            error_message: Error message
            execution_time_ms: Execution time in milliseconds
        
        Returns:
            Error response dictionary
        """
        return {
            "query": query,
            "reasoning": "Error occurred during processing",
            "tools_used": [],
            "results": {},
            "final_answer": f"An error occurred while processing your query: {error_message}",
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "execution_time_ms": round(execution_time_ms, 2),
                "success": False,
                "error": error_message
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the agent and its tools.
        
        Returns:
            Dictionary with agent and tool statistics
        """
        return {
            "agent": {
                "llm_available": self.llm_available,
                "routing_mode": "llm-based" if self.llm_available else "rule-based"
            },
            "tools": {
                "catalog": self.catalog_tool.get_stats(),
                "web_search": self.web_tool.get_stats(),
                "price_analysis": {
                    "llm_available": self.price_tool.llm_available,
                    "default_n_results": self.price_tool.default_n_results
                }
            }
        }


