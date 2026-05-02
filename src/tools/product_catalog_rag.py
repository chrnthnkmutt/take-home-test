"""
Product Catalog RAG (Retrieval-Augmented Generation) tool for the AI Product Research Assistant.

This module provides semantic search capabilities over the product catalog using
vector embeddings and LLM-based answer generation.
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.pipeline.vector_store import VectorStore

logger = logging.getLogger(__name__)


class ProductCatalogRAG:
    """
    RAG-based product catalog search tool.
    
    Provides semantic search over product catalog with metadata filtering,
    LLM-based answer generation, and confidence scoring.
    
    Attributes:
        vector_store: VectorStore instance for semantic search
        llm: LangChain AzureChatOpenAI instance for answer generation
        default_n_results: Default number of results to retrieve
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        persist_directory: str = "./chroma_db",
        collection_name: str = "products",
        azure_api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        temperature: float = 0.7,
        default_n_results: int = 5
    ):
        """
        Initialize the ProductCatalogRAG tool.
        
        Args:
            vector_store: Optional pre-initialized VectorStore instance
            persist_directory: Directory for ChromaDB persistence
            collection_name: Name of the ChromaDB collection
            azure_api_key: Azure OpenAI API key (uses AZURE_OPENAI_API_KEY env var if not provided)
            azure_endpoint: Azure OpenAI endpoint (uses AZURE_OPENAI_ENDPOINT env var if not provided)
            azure_deployment: Azure OpenAI deployment name (uses AZURE_OPENAI_DEPLOYMENT_NAME env var if not provided)
            temperature: LLM temperature for response generation
            default_n_results: Default number of results to return
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
        
        # Initialize LLM with Azure OpenAI
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
                    api_version="2024-02-15-preview",
                    temperature=temperature
                )
                self.llm_available = True
                logger.info(f"LLM initialized with Azure OpenAI deployment: {deployment}")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM: {e}. Will use fallback mode.")
                self.llm_available = False
        else:
            logger.warning("Azure OpenAI credentials not provided. LLM features disabled. "
                         "Set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_DEPLOYMENT_NAME.")
        
        logger.info("ProductCatalogRAG initialized successfully")
    
    def search(
        self,
        query: str,
        n_results: Optional[int] = None,
        category: Optional[str] = None,
        price_range: Optional[Tuple[float, float]] = None,
        brand: Optional[str] = None,
        rating_min: Optional[float] = None,
        stock_status: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search the product catalog using semantic search with optional filters.
        
        Args:
            query: Natural language search query
            n_results: Number of results to return (uses default if not specified)
            category: Filter by product category
            price_range: Tuple of (min_price, max_price) for price filtering
            brand: Filter by brand name
            rating_min: Minimum rating filter (0.0-5.0)
            stock_status: Filter by stock status (e.g., "In Stock", "Low Stock")
        
        Returns:
            Dictionary containing:
                - answer: Natural language answer (if LLM available)
                - products: List of matching products with details
                - confidence: Confidence score (0.0-1.0)
                - sources: List of product IDs used
                - filters_applied: Dictionary of filters that were applied
                - metadata: Additional metadata about the search
        
        Raises:
            ValueError: If query is empty or invalid
            Exception: For other search errors
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        n_results = n_results or self.default_n_results
        
        # Build filters dictionary first (before try block)
        filters_applied = self._build_filters(
            category=category,
            price_range=price_range,
            brand=brand,
            rating_min=rating_min,
            stock_status=stock_status
        )
        
        try:
            # Perform vector search with filters
            search_results = self._perform_search(
                query=query,
                n_results=n_results,
                filters=filters_applied
            )
            
            # Extract and format products
            products = self._format_products(search_results)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(search_results)
            
            # Extract source product IDs
            sources = [p["product_id"] for p in products]
            
            # Generate natural language answer
            answer = self._generate_answer(
                query=query,
                products=products,
                filters_applied=filters_applied
            )
            
            # Build response
            response = {
                "answer": answer,
                "products": products,
                "confidence": confidence,
                "sources": sources,
                "filters_applied": filters_applied,
                "metadata": {
                    "query": query,
                    "n_results_requested": n_results,
                    "n_results_returned": len(products),
                    "timestamp": datetime.utcnow().isoformat(),
                    "llm_used": self.llm_available
                }
            }
            
            logger.info(f"Search completed: query='{query}', results={len(products)}, confidence={confidence:.2f}")
            return response
        
        except ValueError as e:
            logger.error(f"Validation error in search: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return self._create_error_response(query, str(e), filters_applied)
    
    def _build_filters(
        self,
        category: Optional[str] = None,
        price_range: Optional[Tuple[float, float]] = None,
        brand: Optional[str] = None,
        rating_min: Optional[float] = None,
        stock_status: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build filters dictionary for vector store query.
        
        Args:
            category: Product category filter
            price_range: Price range tuple (min, max)
            brand: Brand name filter
            rating_min: Minimum rating filter
            stock_status: Stock status filter
        
        Returns:
            Dictionary of filters to apply
        """
        filters = {}
        
        if category:
            filters["category"] = category
        
        if brand:
            filters["brand"] = brand
        
        if stock_status:
            filters["stock_status"] = stock_status
        
        if price_range:
            min_price, max_price = price_range
            if min_price is not None and max_price is not None:
                filters["price_range"] = {"min": min_price, "max": max_price}
            elif min_price is not None:
                filters["price_range"] = {"min": min_price}
            elif max_price is not None:
                filters["price_range"] = {"max": max_price}
        
        if rating_min is not None:
            filters["rating_min"] = rating_min
        
        return filters
    
    def _perform_search(
        self,
        query: str,
        n_results: int,
        filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform the actual vector store search.
        
        Args:
            query: Search query
            n_results: Number of results to retrieve
            filters: Filters to apply
        
        Returns:
            Raw search results from vector store
        """
        # Extract price range for vector store query
        min_price = None
        max_price = None
        if "price_range" in filters:
            min_price = filters["price_range"].get("min")
            max_price = filters["price_range"].get("max")
        
        # Extract rating min
        rating_min = filters.get("rating_min")
        
        # Perform search with filters
        results = self.vector_store.query_with_filters(
            query_text=query,
            n_results=n_results,
            category=filters.get("category"),
            min_price=min_price,
            max_price=max_price,
            brand=filters.get("brand"),
            min_rating=rating_min,
            stock_status=filters.get("stock_status")
        )
        
        return results
    
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
        documents = search_results["documents"][0] if search_results["documents"] else []
        metadatas = search_results["metadatas"][0] if search_results["metadatas"] else []
        distances = search_results["distances"][0] if search_results["distances"] else []
        
        for i, product_id in enumerate(ids):
            product = {
                "product_id": product_id,
                "description": documents[i] if i < len(documents) else "",
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "similarity_score": 1 - distances[i] if i < len(distances) else 0.0
            }
            
            # Extract key fields from metadata
            if product["metadata"]:
                product["product_name"] = product["metadata"].get("product_name", "Unknown")
                product["category"] = product["metadata"].get("category", "Unknown")
                product["brand"] = product["metadata"].get("brand", "Unknown")
                product["price"] = product["metadata"].get("price", 0.0)
                product["rating"] = product["metadata"].get("rating", 0.0)
                product["stock_status"] = product["metadata"].get("stock_status", "Unknown")
                product["stock_quantity"] = product["metadata"].get("stock_quantity", 0)
            
            products.append(product)
        
        return products
    
    def _calculate_confidence(self, search_results: Dict[str, Any]) -> float:
        """
        Calculate confidence score based on search results.
        
        Args:
            search_results: Raw search results from vector store
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not search_results or "distances" not in search_results:
            return 0.0
        
        distances = search_results["distances"][0] if search_results["distances"] else []
        
        if not distances:
            return 0.0
        
        # Convert distances to similarity scores (1 - distance for cosine)
        similarities = [1 - d for d in distances]
        
        # Calculate average similarity as confidence
        avg_similarity = sum(similarities) / len(similarities)
        
        # Apply scaling to make confidence more meaningful
        # High similarity (>0.8) = high confidence
        # Medium similarity (0.5-0.8) = medium confidence
        # Low similarity (<0.5) = low confidence
        if avg_similarity >= 0.8:
            confidence = 0.8 + (avg_similarity - 0.8) * 1.0  # Scale 0.8-1.0 to 0.8-1.0
        elif avg_similarity >= 0.5:
            confidence = 0.5 + (avg_similarity - 0.5) * 1.0  # Scale 0.5-0.8 to 0.5-0.8
        else:
            confidence = avg_similarity  # Keep as is for low scores
        
        return round(min(max(confidence, 0.0), 1.0), 3)
    
    def _generate_answer(
        self,
        query: str,
        products: List[Dict[str, Any]],
        filters_applied: Dict[str, Any]
    ) -> str:
        """
        Generate natural language answer using LLM or fallback.
        
        Args:
            query: Original user query
            products: List of matching products
            filters_applied: Filters that were applied
        
        Returns:
            Natural language answer string
        """
        if not products:
            return self._generate_no_results_answer(query, filters_applied)
        
        if self.llm_available and self.llm:
            try:
                return self._generate_llm_answer(query, products, filters_applied)
            except Exception as e:
                logger.warning(f"LLM answer generation failed: {e}. Using fallback.")
                return self._generate_fallback_answer(query, products, filters_applied)
        else:
            return self._generate_fallback_answer(query, products, filters_applied)
    
    def _generate_llm_answer(
        self,
        query: str,
        products: List[Dict[str, Any]],
        filters_applied: Dict[str, Any]
    ) -> str:
        """
        Generate answer using LLM.
        
        Args:
            query: User query
            products: Matching products
            filters_applied: Applied filters
        
        Returns:
            LLM-generated answer
        """
        # Build context from products
        product_context = self._build_product_context(products)
        
        # Build filter context
        filter_context = self._build_filter_context(filters_applied)
        
        # Create messages for LangChain
        system_message = SystemMessage(content="""You are a helpful product research assistant. 
Your task is to provide clear, concise, and informative answers about products based on the search results.
Focus on the most relevant products and highlight key features, prices, and ratings.
Be conversational but professional.""")
        
        human_message = HumanMessage(content=f"""User Query: {query}

{filter_context}

Search Results:
{product_context}

Please provide a helpful answer to the user's query based on these search results. 
Include specific product names, prices, and key features. Keep the response concise but informative.""")
        
        # Generate response using LangChain
        response = self.llm.invoke([system_message, human_message])
        # Handle both string and list responses from LLM
        content = response.content if isinstance(response.content, str) else str(response.content[0]) if response.content else ""
        return content.strip()
    
    def _generate_fallback_answer(
        self,
        query: str,
        products: List[Dict[str, Any]],
        filters_applied: Dict[str, Any]
    ) -> str:
        """
        Generate basic answer without LLM.
        
        Args:
            query: User query
            products: Matching products
            filters_applied: Applied filters
        
        Returns:
            Basic formatted answer
        """
        answer_parts = []
        
        # Add intro
        answer_parts.append(f"Found {len(products)} product(s) matching your query.")
        
        # Add filter info if any
        if filters_applied:
            filter_desc = []
            if "category" in filters_applied:
                filter_desc.append(f"category: {filters_applied['category']}")
            if "brand" in filters_applied:
                filter_desc.append(f"brand: {filters_applied['brand']}")
            if "price_range" in filters_applied:
                pr = filters_applied["price_range"]
                if "min" in pr and "max" in pr:
                    filter_desc.append(f"price: ${pr['min']}-${pr['max']}")
                elif "min" in pr:
                    filter_desc.append(f"price: ${pr['min']}+")
                elif "max" in pr:
                    filter_desc.append(f"price: up to ${pr['max']}")
            if "rating_min" in filters_applied:
                filter_desc.append(f"rating: {filters_applied['rating_min']}+ stars")
            
            if filter_desc:
                answer_parts.append(f"Filters applied: {', '.join(filter_desc)}.")
        
        # Add top products
        if products:
            answer_parts.append("\nTop matches:")
            for i, product in enumerate(products[:3], 1):
                name = product.get("product_name", "Unknown")
                price = product.get("price", 0.0)
                rating = product.get("rating", 0.0)
                answer_parts.append(f"{i}. {name} - ${price:.2f} (Rating: {rating}/5.0)")
        
        return "\n".join(answer_parts)
    
    def _generate_no_results_answer(
        self,
        query: str,
        filters_applied: Dict[str, Any]
    ) -> str:
        """
        Generate answer when no results found.
        
        Args:
            query: User query
            filters_applied: Applied filters
        
        Returns:
            No results message
        """
        message = "No products found matching your query."
        
        if filters_applied:
            message += " Try adjusting your filters or search terms."
        
        return message
    
    def _build_product_context(self, products: List[Dict[str, Any]]) -> str:
        """
        Build formatted product context for LLM.
        
        Args:
            products: List of products
        
        Returns:
            Formatted product context string
        """
        context_parts = []
        
        for i, product in enumerate(products, 1):
            name = product.get("product_name", "Unknown")
            price = product.get("price", 0.0)
            rating = product.get("rating", 0.0)
            category = product.get("category", "Unknown")
            brand = product.get("brand", "Unknown")
            description = product.get("description", "")
            stock_status = product.get("stock_status", "Unknown")
            
            context = f"""
Product {i}:
- Name: {name}
- Brand: {brand}
- Category: {category}
- Price: ${price:.2f}
- Rating: {rating}/5.0
- Stock: {stock_status}
- Description: {description[:200]}...
"""
            context_parts.append(context.strip())
        
        return "\n\n".join(context_parts)
    
    def _build_filter_context(self, filters_applied: Dict[str, Any]) -> str:
        """
        Build formatted filter context.
        
        Args:
            filters_applied: Applied filters
        
        Returns:
            Formatted filter context string
        """
        if not filters_applied:
            return "No filters applied."
        
        filter_parts = ["Filters Applied:"]
        
        if "category" in filters_applied:
            filter_parts.append(f"- Category: {filters_applied['category']}")
        if "brand" in filters_applied:
            filter_parts.append(f"- Brand: {filters_applied['brand']}")
        if "price_range" in filters_applied:
            pr = filters_applied["price_range"]
            if "min" in pr and "max" in pr:
                filter_parts.append(f"- Price Range: ${pr['min']:.2f} - ${pr['max']:.2f}")
            elif "min" in pr:
                filter_parts.append(f"- Minimum Price: ${pr['min']:.2f}")
            elif "max" in pr:
                filter_parts.append(f"- Maximum Price: ${pr['max']:.2f}")
        if "rating_min" in filters_applied:
            filter_parts.append(f"- Minimum Rating: {filters_applied['rating_min']}/5.0")
        if "stock_status" in filters_applied:
            filter_parts.append(f"- Stock Status: {filters_applied['stock_status']}")
        
        return "\n".join(filter_parts)
    
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
            "answer": f"An error occurred while searching: {error_message}",
            "products": [],
            "confidence": 0.0,
            "sources": [],
            "filters_applied": filters_applied,
            "metadata": {
                "query": query,
                "n_results_requested": self.default_n_results,
                "n_results_returned": 0,
                "timestamp": datetime.utcnow().isoformat(),
                "llm_used": False,
                "error": error_message
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG tool and vector store.
        
        Returns:
            Dictionary with statistics
        """
        stats = self.vector_store.get_collection_stats()
        stats["vector_store_initialized"] = self.vector_store.is_initialized()
        stats["total_documents"] = stats.get("document_count", 0)
        stats["llm_available"] = self.llm_available
        stats["default_n_results"] = self.default_n_results
        
        return stats


