"""
Price Analysis & Recommendation tool for the AI Product Research Assistant.

This module provides price analysis capabilities including margin calculations,
markup analysis, and profit calculations using deterministic functions.
LLM is used only for formatting results and providing insights.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from pydantic import SecretStr
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.pipeline.vector_store import VectorStore

logger = logging.getLogger(__name__)


# Deterministic calculation functions
def calculate_margin(price: float, cost: float) -> float:
    """
    Calculate profit margin percentage.
    
    Formula: ((price - cost) / price) * 100
    
    Args:
        price: Product selling price
        cost: Product cost
    
    Returns:
        Profit margin as a percentage (0-100)
    
    Raises:
        ValueError: If price is zero or negative
        ZeroDivisionError: If price is zero
    """
    if price <= 0:
        raise ValueError(f"Price must be positive, got {price}")
    
    if cost < 0:
        raise ValueError(f"Cost cannot be negative, got {cost}")
    
    margin = ((price - cost) / price) * 100
    return round(margin, 2)


def calculate_markup(price: float, cost: float) -> float:
    """
    Calculate markup percentage.
    
    Formula: ((price - cost) / cost) * 100
    
    Args:
        price: Product selling price
        cost: Product cost
    
    Returns:
        Markup as a percentage
    
    Raises:
        ValueError: If cost is zero or negative
        ZeroDivisionError: If cost is zero
    """
    if cost <= 0:
        raise ValueError(f"Cost must be positive, got {cost}")
    
    if price < 0:
        raise ValueError(f"Price cannot be negative, got {price}")
    
    markup = ((price - cost) / cost) * 100
    return round(markup, 2)


def calculate_profit(price: float, cost: float) -> float:
    """
    Calculate absolute profit.
    
    Formula: price - cost
    
    Args:
        price: Product selling price
        cost: Product cost
    
    Returns:
        Absolute profit amount
    
    Raises:
        ValueError: If price or cost is negative
    """
    if price < 0:
        raise ValueError(f"Price cannot be negative, got {price}")
    
    if cost < 0:
        raise ValueError(f"Cost cannot be negative, got {cost}")
    
    profit = price - cost
    return round(profit, 2)


class PriceAnalysisTool:
    """
    Price Analysis tool for analyzing product pricing and margins.
    
    Uses deterministic calculation functions for all mathematical operations.
    LLM is used only for formatting results and providing insights.
    
    Attributes:
        vector_store: VectorStore instance for querying products
        llm: LangChain AzureChatOpenAI instance for insights
        default_n_results: Default number of products to analyze
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        persist_directory: str = "./chroma_db",
        collection_name: str = "products",
        azure_api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        temperature: float = 0.3,
        default_n_results: int = 10
    ):
        """
        Initialize the PriceAnalysisTool.
        
        Args:
            vector_store: Optional pre-initialized VectorStore instance
            persist_directory: Directory for ChromaDB persistence
            collection_name: Name of the ChromaDB collection
            azure_api_key: Azure OpenAI API key (uses env var AZURE_OPENAI_API_KEY if not provided)
            azure_endpoint: Azure OpenAI endpoint (uses env var AZURE_OPENAI_ENDPOINT if not provided)
            azure_deployment: Azure OpenAI deployment name (uses env var AZURE_OPENAI_DEPLOYMENT_NAME if not provided)
            temperature: LLM temperature (lower for more deterministic insights)
            default_n_results: Default number of products to analyze
        """
        # Initialize vector store
        if vector_store:
            self.vector_store = vector_store
        else:
            self.vector_store = VectorStore(
                persist_directory=persist_directory,
                collection_name=collection_name
            )
            self.vector_store.get_or_create_collection()
        
        self.default_n_results = default_n_results
        
        # Initialize LLM for insights only
        import os
        api_key = azure_api_key or os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.llm = None
        self.llm_available = False
        
        if api_key and endpoint and deployment:
            try:
                self.llm = AzureChatOpenAI(
                    azure_endpoint=endpoint,
                    azure_deployment=deployment,
                    api_key=SecretStr(api_key),
                    api_version="2024-02-15-preview",
                    temperature=temperature
                )
                self.llm_available = True
                logger.info(f"LLM initialized with Azure OpenAI deployment: {deployment}")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM: {e}. Will use fallback mode.")
                self.llm_available = False
        else:
            logger.warning("Azure OpenAI credentials not provided. LLM features disabled.")
        
        logger.info("PriceAnalysisTool initialized successfully")
    
    def analyze(
        self,
        query: str,
        n_results: Optional[int] = None,
        category: Optional[str] = None,
        brand: Optional[str] = None,
        price_range: Optional[Tuple[float, float]] = None,
        margin_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Analyze product pricing and margins based on query.
        
        Supports queries like:
        - "Which products have the lowest profit margins?"
        - "Calculate average margin for Electronics category"
        - "Show me products with margins below 40%"
        
        Args:
            query: Natural language query about pricing/margins
            n_results: Number of products to analyze
            category: Filter by product category
            brand: Filter by brand name
            price_range: Tuple of (min_price, max_price)
            margin_threshold: Filter products by margin threshold
        
        Returns:
            Dictionary containing:
                - analysis: LLM-generated insights and summary
                - calculations: Aggregate calculations (deterministic)
                - products: List of products with calculated metrics
                - metadata: Query metadata and filters applied
        
        Raises:
            ValueError: If query is empty or invalid
            Exception: For other analysis errors
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        n_results = n_results or self.default_n_results
        
        # Build filters
        filters_applied = self._build_filters(
            category=category,
            brand=brand,
            price_range=price_range
        )
        
        try:
            # Query products from vector store
            products = self._query_products(
                query=query,
                n_results=n_results,
                filters=filters_applied
            )
            
            if not products:
                return self._create_no_results_response(query, filters_applied)
            
            # Calculate metrics for each product (deterministic)
            products_with_metrics = self._calculate_product_metrics(products)
            
            # Apply margin threshold filter if specified
            if margin_threshold is not None:
                products_with_metrics = [
                    p for p in products_with_metrics
                    if p.get("margin") is not None and p["margin"] <= margin_threshold
                ]
                filters_applied["margin_threshold"] = margin_threshold
            
            if not products_with_metrics:
                return self._create_no_results_response(query, filters_applied)
            
            # Calculate aggregate statistics (deterministic)
            calculations = self._calculate_aggregates(products_with_metrics)
            
            # Generate LLM insights (NOT for calculations)
            analysis = self._generate_analysis(
                query=query,
                products=products_with_metrics,
                calculations=calculations,
                filters_applied=filters_applied
            )
            
            # Build response
            response = {
                "analysis": analysis,
                "calculations": calculations,
                "products": products_with_metrics,
                "metadata": {
                    "query": query,
                    "filters_applied": filters_applied,
                    "timestamp": datetime.utcnow().isoformat(),
                    "n_products_analyzed": len(products_with_metrics),
                    "llm_used": self.llm_available
                }
            }
            
            logger.info(f"Price analysis completed: query='{query}', products={len(products_with_metrics)}")
            return response
        
        except ValueError as e:
            logger.error(f"Validation error in analysis: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during price analysis: {e}")
            return self._create_error_response(query, str(e), filters_applied)
    
    def _build_filters(
        self,
        category: Optional[str] = None,
        brand: Optional[str] = None,
        price_range: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """
        Build filters dictionary for vector store query.
        
        Args:
            category: Product category filter
            brand: Brand name filter
            price_range: Price range tuple (min, max)
        
        Returns:
            Dictionary of filters to apply
        """
        filters = {}
        
        if category:
            filters["category"] = category
        
        if brand:
            filters["brand"] = brand
        
        if price_range:
            min_price, max_price = price_range
            if min_price is not None and max_price is not None:
                filters["price_range"] = {"min": min_price, "max": max_price}
            elif min_price is not None:
                filters["price_range"] = {"min": min_price}
            elif max_price is not None:
                filters["price_range"] = {"max": max_price}
        
        return filters
    
    def _query_products(
        self,
        query: str,
        n_results: int,
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Query products from vector store with filters.
        
        Args:
            query: Search query
            n_results: Number of results to retrieve
            filters: Filters to apply
        
        Returns:
            List of products with metadata
        """
        # Extract price range for vector store query
        min_price = None
        max_price = None
        if "price_range" in filters:
            min_price = filters["price_range"].get("min")
            max_price = filters["price_range"].get("max")
        
        # Perform search with filters
        results = self.vector_store.query_with_filters(
            query_text=query,
            n_results=n_results,
            category=filters.get("category"),
            min_price=min_price,
            max_price=max_price,
            brand=filters.get("brand")
        )
        
        # Format products
        products = self._format_products(results)
        return products
    
    def _format_products(self, search_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Format raw search results into structured product list.
        
        Args:
            search_results: Raw results from vector store
        
        Returns:
            List of formatted product dictionaries
        """
        products = []
        
        if not search_results or "ids" not in search_results:
            return products
        
        ids = search_results["ids"][0] if search_results["ids"] else []
        metadatas = search_results["metadatas"][0] if search_results["metadatas"] else []
        
        for i, product_id in enumerate(ids):
            metadata = metadatas[i] if i < len(metadatas) else {}
            
            product = {
                "product_id": product_id,
                "name": metadata.get("product_name", "Unknown"),
                "category": metadata.get("category", "Unknown"),
                "brand": metadata.get("brand", "Unknown"),
                "price": metadata.get("price", 0.0),
                "cost": metadata.get("cost"),  # May be None
                "rating": metadata.get("rating", 0.0),
                "stock_status": metadata.get("stock_status", "Unknown")
            }
            
            products.append(product)
        
        return products
    
    def _calculate_product_metrics(
        self,
        products: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Calculate pricing metrics for each product using deterministic functions.
        
        Args:
            products: List of products
        
        Returns:
            List of products with calculated metrics
        """
        products_with_metrics = []
        
        for product in products:
            price = product.get("price", 0.0)
            cost = product.get("cost")
            
            # Create a copy to avoid modifying original
            product_with_metrics = product.copy()
            
            # Calculate metrics only if both price and cost are available
            if cost is not None and price > 0 and cost >= 0:
                try:
                    product_with_metrics["margin"] = calculate_margin(price, cost)
                    product_with_metrics["markup"] = calculate_markup(price, cost)
                    product_with_metrics["profit"] = calculate_profit(price, cost)
                except (ValueError, ZeroDivisionError) as e:
                    logger.warning(f"Error calculating metrics for product {product.get('product_id')}: {e}")
                    product_with_metrics["margin"] = None
                    product_with_metrics["markup"] = None
                    product_with_metrics["profit"] = None
                    product_with_metrics["calculation_error"] = str(e)
            else:
                # Missing cost data
                product_with_metrics["margin"] = None
                product_with_metrics["markup"] = None
                product_with_metrics["profit"] = None
                if cost is None:
                    product_with_metrics["calculation_error"] = "Cost data not available"
            
            products_with_metrics.append(product_with_metrics)
        
        return products_with_metrics
    
    def _calculate_aggregates(
        self,
        products: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate aggregate statistics using deterministic functions.
        
        Args:
            products: List of products with calculated metrics
        
        Returns:
            Dictionary of aggregate calculations
        """
        # Filter products with valid metrics
        valid_products = [
            p for p in products
            if p.get("margin") is not None and p.get("markup") is not None
        ]
        
        if not valid_products:
            return {
                "total_products": len(products),
                "products_with_cost_data": 0,
                "average_margin": None,
                "average_markup": None,
                "total_profit": None,
                "min_margin": None,
                "max_margin": None,
                "min_markup": None,
                "max_markup": None
            }
        
        margins = [p["margin"] for p in valid_products]
        markups = [p["markup"] for p in valid_products]
        profits = [p["profit"] for p in valid_products]
        
        return {
            "total_products": len(products),
            "products_with_cost_data": len(valid_products),
            "average_margin": round(sum(margins) / len(margins), 2),
            "average_markup": round(sum(markups) / len(markups), 2),
            "total_profit": round(sum(profits), 2),
            "min_margin": round(min(margins), 2),
            "max_margin": round(max(margins), 2),
            "min_markup": round(min(markups), 2),
            "max_markup": round(max(markups), 2)
        }
    
    def _generate_analysis(
        self,
        query: str,
        products: List[Dict[str, Any]],
        calculations: Dict[str, Any],
        filters_applied: Dict[str, Any]
    ) -> str:
        """
        Generate natural language analysis using LLM or fallback.
        
        LLM is used ONLY for formatting and insights, NOT for calculations.
        
        Args:
            query: Original user query
            products: Products with calculated metrics
            calculations: Aggregate calculations (already computed)
            filters_applied: Filters that were applied
        
        Returns:
            Natural language analysis string
        """
        if self.llm_available and self.llm:
            try:
                return self._generate_llm_analysis(query, products, calculations, filters_applied)
            except Exception as e:
                logger.warning(f"LLM analysis generation failed: {e}. Using fallback.")
                return self._generate_fallback_analysis(query, products, calculations, filters_applied)
        else:
            return self._generate_fallback_analysis(query, products, calculations, filters_applied)
    
    def _generate_llm_analysis(
        self,
        query: str,
        products: List[Dict[str, Any]],
        calculations: Dict[str, Any],
        filters_applied: Dict[str, Any]
    ) -> str:
        """
        Generate analysis using LLM for insights and formatting only.
        
        Args:
            query: User query
            products: Products with metrics
            calculations: Pre-calculated aggregates
            filters_applied: Applied filters
        
        Returns:
            LLM-generated analysis
        """
        # Build context with pre-calculated data
        context = self._build_analysis_context(products, calculations, filters_applied)
        
        system_message = SystemMessage(content="""You are a pricing analysis expert assistant.
Your task is to provide clear, actionable insights about product pricing and margins.
IMPORTANT: All calculations have already been performed. Do NOT recalculate any numbers.
Use the provided calculations to generate insights, recommendations, and observations.
Focus on trends, comparisons, and strategic recommendations.""")
        
        human_message = HumanMessage(content=f"""User Query: {query}

{context}

Please provide a comprehensive analysis with:
1. Summary of the pricing situation
2. Key insights and observations
3. Recommendations for pricing strategy
4. Any notable trends or outliers

Remember: All numbers are already calculated. Focus on interpretation and insights.""")
        
        response = self.llm.invoke([system_message, human_message])
        # Handle response content
        content = response.content if isinstance(response.content, str) else str(response.content)
        return content.strip()
    
    def _generate_fallback_analysis(
        self,
        query: str,
        products: List[Dict[str, Any]],
        calculations: Dict[str, Any],
        filters_applied: Dict[str, Any]
    ) -> str:
        """
        Generate basic analysis without LLM.
        
        Args:
            query: User query
            products: Products with metrics
            calculations: Pre-calculated aggregates
            filters_applied: Applied filters
        
        Returns:
            Basic formatted analysis
        """
        analysis_parts = []
        
        # Summary
        total = calculations["total_products"]
        with_cost = calculations["products_with_cost_data"]
        analysis_parts.append(f"Price Analysis Summary:")
        analysis_parts.append(f"- Analyzed {total} product(s), {with_cost} with cost data available")
        
        # Filters
        if filters_applied:
            filter_desc = []
            if "category" in filters_applied:
                filter_desc.append(f"Category: {filters_applied['category']}")
            if "brand" in filters_applied:
                filter_desc.append(f"Brand: {filters_applied['brand']}")
            if "price_range" in filters_applied:
                pr = filters_applied["price_range"]
                if "min" in pr and "max" in pr:
                    filter_desc.append(f"Price: ${pr['min']}-${pr['max']}")
            if filter_desc:
                analysis_parts.append(f"- Filters: {', '.join(filter_desc)}")
        
        # Aggregate metrics
        if with_cost > 0:
            analysis_parts.append(f"\nAggregate Metrics:")
            analysis_parts.append(f"- Average Margin: {calculations['average_margin']}%")
            analysis_parts.append(f"- Average Markup: {calculations['average_markup']}%")
            analysis_parts.append(f"- Total Profit: ${calculations['total_profit']}")
            analysis_parts.append(f"- Margin Range: {calculations['min_margin']}% - {calculations['max_margin']}%")
            
            # Top/bottom performers
            valid_products = [p for p in products if p.get("margin") is not None]
            if valid_products:
                sorted_by_margin = sorted(valid_products, key=lambda x: x["margin"])
                
                analysis_parts.append(f"\nLowest Margin Product:")
                lowest = sorted_by_margin[0]
                analysis_parts.append(f"- {lowest['name']}: {lowest['margin']}% margin, ${lowest['price']} price")
                
                analysis_parts.append(f"\nHighest Margin Product:")
                highest = sorted_by_margin[-1]
                analysis_parts.append(f"- {highest['name']}: {highest['margin']}% margin, ${highest['price']} price")
        else:
            analysis_parts.append("\nNo cost data available for margin calculations.")
        
        return "\n".join(analysis_parts)
    
    def _build_analysis_context(
        self,
        products: List[Dict[str, Any]],
        calculations: Dict[str, Any],
        filters_applied: Dict[str, Any]
    ) -> str:
        """
        Build formatted context for LLM analysis.
        
        Args:
            products: Products with metrics
            calculations: Pre-calculated aggregates
            filters_applied: Applied filters
        
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Aggregate calculations
        context_parts.append("Aggregate Calculations (Pre-computed):")
        context_parts.append(f"- Total Products: {calculations['total_products']}")
        context_parts.append(f"- Products with Cost Data: {calculations['products_with_cost_data']}")
        
        if calculations['products_with_cost_data'] > 0:
            context_parts.append(f"- Average Margin: {calculations['average_margin']}%")
            context_parts.append(f"- Average Markup: {calculations['average_markup']}%")
            context_parts.append(f"- Total Profit: ${calculations['total_profit']}")
            context_parts.append(f"- Margin Range: {calculations['min_margin']}% to {calculations['max_margin']}%")
            context_parts.append(f"- Markup Range: {calculations['min_markup']}% to {calculations['max_markup']}%")
        
        # Filters
        if filters_applied:
            context_parts.append("\nFilters Applied:")
            for key, value in filters_applied.items():
                context_parts.append(f"- {key}: {value}")
        
        # Top products by margin
        valid_products = [p for p in products if p.get("margin") is not None]
        if valid_products:
            sorted_by_margin = sorted(valid_products, key=lambda x: x["margin"])
            
            context_parts.append("\nTop 5 Products by Margin:")
            for i, product in enumerate(sorted_by_margin[-5:][::-1], 1):
                context_parts.append(
                    f"{i}. {product['name']}: Margin={product['margin']}%, "
                    f"Markup={product['markup']}%, Price=${product['price']}, Profit=${product['profit']}"
                )
            
            context_parts.append("\nBottom 5 Products by Margin:")
            for i, product in enumerate(sorted_by_margin[:5], 1):
                context_parts.append(
                    f"{i}. {product['name']}: Margin={product['margin']}%, "
                    f"Markup={product['markup']}%, Price=${product['price']}, Profit=${product['profit']}"
                )
        
        return "\n".join(context_parts)
    
    def _create_no_results_response(
        self,
        query: str,
        filters_applied: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create response when no products found.
        
        Args:
            query: Original query
            filters_applied: Filters that were applied
        
        Returns:
            No results response dictionary
        """
        return {
            "analysis": "No products found matching your criteria. Try adjusting your filters or search terms.",
            "calculations": {
                "total_products": 0,
                "products_with_cost_data": 0,
                "average_margin": None,
                "average_markup": None,
                "total_profit": None,
                "min_margin": None,
                "max_margin": None
            },
            "products": [],
            "metadata": {
                "query": query,
                "filters_applied": filters_applied,
                "timestamp": datetime.utcnow().isoformat(),
                "n_products_analyzed": 0,
                "llm_used": False
            }
        }
    
    def _create_error_response(
        self,
        query: str,
        error_message: str,
        filters_applied: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create error response structure.
        
        Args:
            query: Original query
            error_message: Error message
            filters_applied: Filters that were applied
        
        Returns:
            Error response dictionary
        """
        return {
            "analysis": f"An error occurred during price analysis: {error_message}",
            "calculations": {
                "total_products": 0,
                "products_with_cost_data": 0,
                "average_margin": None,
                "average_markup": None,
                "total_profit": None,
                "min_margin": None,
                "max_margin": None
            },
            "products": [],
            "metadata": {
                "query": query,
                "filters_applied": filters_applied,
                "timestamp": datetime.utcnow().isoformat(),
                "n_products_analyzed": 0,
                "llm_used": False,
                "error": error_message
            }
        }


