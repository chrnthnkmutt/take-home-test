"""
Web Search tool for the AI Product Research Assistant.

This module provides web search capabilities using multiple search APIs with fallback support.
Supports Tavily API with mock implementation when API keys are unavailable.
"""

import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import requests

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class WebSearchTool:
    """
    Web search tool with multiple API support and fallback.
    
    Provides web search capabilities with support for Tavily API and mock implementation.
    Returns structured results with answer summary, search results, sources, and metadata.
    
    Attributes:
        api_key: Tavily API key
        llm: LangChain AzureChatOpenAI instance for answer generation
        default_n_results: Default number of results to retrieve
        api_available: Whether a real API is available
    """
    
    def __init__(
        self,
        tavily_api_key: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        temperature: float = 0.7,
        default_n_results: int = 5
    ):
        """
        Initialize the WebSearchTool.
        
        Args:
            tavily_api_key: Tavily API key (uses env var if not provided)
            azure_api_key: Azure OpenAI API key (uses env var AZURE_OPENAI_API_KEY if not provided)
            azure_endpoint: Azure OpenAI endpoint (uses env var AZURE_OPENAI_ENDPOINT if not provided)
            azure_deployment: Azure OpenAI deployment name (uses env var AZURE_OPENAI_DEPLOYMENT_NAME if not provided)
            temperature: LLM temperature for response generation
            default_n_results: Default number of results to return
        """
        self.default_n_results = default_n_results
        
        # Initialize Tavily API
        self.api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        self.api_available = bool(self.api_key)
        
        if self.api_available:
            logger.info("Tavily API key found. Web search will use Tavily API.")
        else:
            logger.warning("No Tavily API key found. Web search will use mock implementation.")
        
        # Initialize LLM for answer generation
        # Set environment variables if provided as parameters
        if azure_api_key:
            os.environ["AZURE_OPENAI_API_KEY"] = azure_api_key
        if azure_endpoint:
            os.environ["AZURE_OPENAI_ENDPOINT"] = azure_endpoint
        if azure_deployment:
            os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = azure_deployment
            
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
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
            logger.warning("Azure OpenAI credentials not fully provided. LLM features disabled.")
        
        logger.info("WebSearchTool initialized successfully")
    
    def search(
        self,
        query: str,
        n_results: Optional[int] = None,
        search_depth: str = "basic"
    ) -> Dict[str, Any]:
        """
        Search the web using available APIs or mock implementation.
        
        Args:
            query: Natural language search query
            n_results: Number of results to return (uses default if not specified)
            search_depth: Search depth - "basic" or "advanced" (for Tavily API)
        
        Returns:
            Dictionary containing:
                - answer: Summary of search results
                - results: List of search results with title, url, snippet, relevance_score
                - sources: List of URLs used
                - metadata: Additional metadata about the search
        
        Raises:
            ValueError: If query is empty or invalid
            Exception: For other search errors
        
        Example:
            >>> tool = WebSearchTool()
            >>> results = tool.search("What is the current market price for noise-cancelling headphones?")
            >>> print(results["answer"])
            >>> for result in results["results"]:
            ...     print(f"{result['title']}: {result['url']}")
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        n_results = n_results or self.default_n_results
        
        try:
            # Perform search using available API or mock
            if self.api_available:
                search_results = self._search_tavily(query, n_results, search_depth)
            else:
                search_results = self._search_mock(query, n_results)
            
            # Extract sources
            sources = [result["url"] for result in search_results]
            
            # Generate answer summary
            answer = self._generate_answer(query, search_results)
            
            # Build response
            response = {
                "answer": answer,
                "results": search_results,
                "sources": sources,
                "metadata": {
                    "query": query,
                    "api_used": "tavily" if self.api_available else "mock",
                    "n_results_requested": n_results,
                    "n_results_returned": len(search_results),
                    "timestamp": datetime.utcnow().isoformat(),
                    "llm_used": self.llm_available
                }
            }
            
            logger.info(f"Search completed: query='{query}', results={len(search_results)}, api={'tavily' if self.api_available else 'mock'}")
            return response
        
        except ValueError as e:
            logger.error(f"Validation error in search: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return self._create_error_response(query, str(e))
    
    def _search_tavily(
        self,
        query: str,
        n_results: int,
        search_depth: str
    ) -> List[Dict[str, Any]]:
        """
        Search using Tavily API.
        
        Args:
            query: Search query
            n_results: Number of results to retrieve
            search_depth: Search depth ("basic" or "advanced")
        
        Returns:
            List of search results
        
        Raises:
            Exception: If API request fails
        """
        try:
            url = "https://api.tavily.com/search"
            
            payload = {
                "api_key": self.api_key,
                "query": query,
                "max_results": n_results,
                "search_depth": search_depth,
                "include_answer": True,
                "include_raw_content": False
            }
            
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Format results
            results = []
            for item in data.get("results", []):
                result = {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", ""),
                    "relevance_score": item.get("score", 0.5)
                }
                results.append(result)
            
            logger.info(f"Tavily API returned {len(results)} results")
            return results
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Tavily API request failed: {e}")
            logger.warning("Falling back to mock implementation")
            return self._search_mock(query, n_results)
        except Exception as e:
            logger.error(f"Error processing Tavily response: {e}")
            logger.warning("Falling back to mock implementation")
            return self._search_mock(query, n_results)
    
    def _search_mock(self, query: str, n_results: int) -> List[Dict[str, Any]]:
        """
        Mock search implementation with realistic sample data.
        
        This provides realistic product research data when no API is available.
        The mock data is contextually relevant to common product research queries.
        
        Args:
            query: Search query
            n_results: Number of results to return
        
        Returns:
            List of mock search results
        """
        logger.info(f"Using mock search for query: '{query}'")
        
        # Determine query category for relevant mock data
        query_lower = query.lower()
        
        # Mock data for different product categories
        mock_data = []
        
        if any(term in query_lower for term in ["headphone", "audio", "noise-cancelling", "sony", "bose"]):
            mock_data = [
                {
                    "title": "Sony WH-1000XM5 Review: Best Noise-Cancelling Headphones 2024",
                    "url": "https://www.techradar.com/reviews/sony-wh-1000xm5",
                    "snippet": "The Sony WH-1000XM5 are the best noise-cancelling headphones you can buy right now. With industry-leading ANC, exceptional sound quality, and 30-hour battery life, they're worth the $399 price tag. The new design is sleeker and more comfortable for long listening sessions.",
                    "relevance_score": 0.95
                },
                {
                    "title": "Bose QuietComfort Ultra vs Sony WH-1000XM5: Which Should You Buy?",
                    "url": "https://www.cnet.com/tech/mobile/bose-qc-ultra-vs-sony-wh-1000xm5/",
                    "snippet": "Both headphones offer excellent noise cancellation, but the Sony WH-1000XM5 ($399) edges out the Bose QuietComfort Ultra ($429) in sound quality and battery life. However, Bose wins on comfort and build quality. Current market prices show both models frequently on sale for $50-70 off.",
                    "relevance_score": 0.92
                },
                {
                    "title": "Best Noise-Cancelling Headphones 2024: Top Picks Under $400",
                    "url": "https://www.wirecutter.com/reviews/best-noise-cancelling-headphones/",
                    "snippet": "Our top picks for noise-cancelling headphones include: 1) Sony WH-1000XM5 ($399) - Best overall, 2) Bose QuietComfort Ultra ($429) - Best comfort, 3) Apple AirPods Max ($549) - Best for Apple users, 4) Sennheiser Momentum 4 ($379) - Best battery life at 60 hours.",
                    "relevance_score": 0.89
                },
                {
                    "title": "Sony WH-1000XM5 Price Drop: Now $299 at Amazon",
                    "url": "https://www.amazon.com/deals/sony-wh-1000xm5",
                    "snippet": "Limited time deal: Sony WH-1000XM5 wireless noise-cancelling headphones now available for $299, down from $399. This is the lowest price we've seen for these premium headphones. Features include 30-hour battery, multipoint connectivity, and exceptional ANC performance.",
                    "relevance_score": 0.88
                },
                {
                    "title": "Noise-Cancelling Headphones Market Analysis 2024",
                    "url": "https://www.marketresearch.com/audio-market-2024",
                    "snippet": "The premium noise-cancelling headphones market is dominated by Sony (35% market share) and Bose (28%). Average selling price has stabilized around $350-400 for flagship models. Consumer demand remains strong with 15% YoY growth expected through 2025.",
                    "relevance_score": 0.82
                }
            ]
        
        elif any(term in query_lower for term in ["fitness", "exercise", "workout", "gym", "home fitness"]):
            mock_data = [
                {
                    "title": "Best Home Fitness Equipment 2024: Top Trending Products",
                    "url": "https://www.menshealth.com/fitness/best-home-gym-equipment",
                    "snippet": "Trending home fitness equipment includes: 1) Adjustable dumbbells ($299-499), 2) Smart rowing machines ($1,000-2,500), 3) Compact treadmills ($500-1,200), 4) Resistance band sets ($30-80), and 5) Yoga mats with alignment guides ($50-100). Sales have increased 40% since 2023.",
                    "relevance_score": 0.94
                },
                {
                    "title": "Peloton vs NordicTrack: Which Home Fitness System is Worth It?",
                    "url": "https://www.cnet.com/health/fitness/peloton-vs-nordictrack/",
                    "snippet": "Peloton Bike+ ($2,495) and NordicTrack S22i ($1,999) are top contenders in connected fitness. Both offer live classes and competitive features. NordicTrack provides better value with included iFit membership, while Peloton has a larger community and more class variety.",
                    "relevance_score": 0.91
                },
                {
                    "title": "Home Gym Equipment Market Trends: What's Hot in 2024",
                    "url": "https://www.fitnessmagazine.com/home-gym-trends-2024",
                    "snippet": "Compact, multi-functional equipment is trending. Top sellers include: Bowflex SelectTech dumbbells ($399), TRX suspension trainers ($179), and foldable treadmills ($600-900). Smart fitness mirrors have seen 60% growth, with prices ranging from $1,200-1,500.",
                    "relevance_score": 0.87
                },
                {
                    "title": "Best Budget Home Fitness Equipment Under $300",
                    "url": "https://www.wirecutter.com/reviews/best-budget-home-gym/",
                    "snippet": "Quality home fitness doesn't require breaking the bank. Our top budget picks: 1) Resistance band set ($45), 2) Adjustable kettlebell ($89), 3) Yoga mat and blocks bundle ($65), 4) Jump rope ($25), 5) Foam roller ($35). Total investment under $300 for a complete home gym.",
                    "relevance_score": 0.85
                },
                {
                    "title": "Smart Fitness Equipment: Connected Devices Revolutionizing Home Workouts",
                    "url": "https://www.techcrunch.com/smart-fitness-equipment-2024",
                    "snippet": "Smart fitness equipment market expected to reach $12B by 2025. Popular products include Mirror ($1,495), Tonal ($3,995), and Tempo Studio ($2,495). These AI-powered systems offer personalized training, form correction, and progress tracking.",
                    "relevance_score": 0.83
                }
            ]
        
        elif any(term in query_lower for term in ["laptop", "computer", "macbook", "dell", "hp"]):
            mock_data = [
                {
                    "title": "Best Laptops 2024: Top Picks for Every Budget",
                    "url": "https://www.laptopmag.com/best-laptops-2024",
                    "snippet": "Our top laptop picks: 1) MacBook Air M3 ($1,099) - Best overall, 2) Dell XPS 13 ($999) - Best Windows laptop, 3) Lenovo ThinkPad X1 Carbon ($1,399) - Best for business, 4) ASUS ROG Zephyrus ($1,799) - Best gaming. All models feature excellent performance and battery life.",
                    "relevance_score": 0.93
                },
                {
                    "title": "MacBook Air M3 vs MacBook Pro: Which Should You Buy?",
                    "url": "https://www.macrumors.com/guide/macbook-air-vs-pro/",
                    "snippet": "The MacBook Air M3 ($1,099) offers incredible value with 18-hour battery life and fanless design. MacBook Pro 14-inch ($1,999) provides more power for professionals. For most users, the Air is sufficient and $900 cheaper than the Pro.",
                    "relevance_score": 0.90
                },
                {
                    "title": "Laptop Market Prices: Best Deals and Trends Q2 2024",
                    "url": "https://www.pcworld.com/laptop-deals-2024",
                    "snippet": "Average laptop prices have decreased 8% YoY. Budget laptops ($400-600) now offer solid performance. Mid-range ($800-1,200) provides best value. Premium models ($1,500+) feature OLED displays and high-end specs. Best time to buy is during back-to-school sales in July-August.",
                    "relevance_score": 0.88
                },
                {
                    "title": "Dell XPS 13 Plus Review: Premium Design Meets Performance",
                    "url": "https://www.theverge.com/reviews/dell-xps-13-plus",
                    "snippet": "The Dell XPS 13 Plus ($1,299) features a stunning design with edge-to-edge keyboard and invisible trackpad. Intel Core Ultra 7 processor delivers excellent performance. 13.4-inch OLED display option available for $200 more. Battery life reaches 12 hours with FHD+ display.",
                    "relevance_score": 0.86
                },
                {
                    "title": "Budget Laptops That Don't Compromise: Best Under $700",
                    "url": "https://www.cnet.com/tech/computing/best-budget-laptops/",
                    "snippet": "Quality budget laptops include: Acer Aspire 5 ($549), HP Pavilion 15 ($629), and Lenovo IdeaPad 3 ($499). All feature 8GB RAM, 256GB SSD, and 1080p displays. Perfect for students and everyday computing tasks.",
                    "relevance_score": 0.84
                }
            ]
        
        else:
            # Generic product research mock data
            mock_data = [
                {
                    "title": "Product Research Guide: Finding the Best Deals in 2024",
                    "url": "https://www.consumerreports.org/product-research-guide",
                    "snippet": "Comprehensive guide to product research: compare prices across retailers, read verified reviews, check price history, and wait for seasonal sales. Best times to buy: Black Friday (November), Prime Day (July), and end-of-season clearances.",
                    "relevance_score": 0.90
                },
                {
                    "title": "How to Compare Products: Expert Tips for Smart Shopping",
                    "url": "https://www.wirecutter.com/blog/how-to-compare-products/",
                    "snippet": "Key factors for product comparison: 1) Read multiple reviews from trusted sources, 2) Compare specifications side-by-side, 3) Check warranty and return policies, 4) Consider total cost of ownership, 5) Look for certified refurbished options to save 20-30%.",
                    "relevance_score": 0.87
                },
                {
                    "title": "Best Price Comparison Tools and Websites 2024",
                    "url": "https://www.pcmag.com/picks/best-price-comparison-sites",
                    "snippet": "Top price comparison tools: Google Shopping, CamelCamelCamel (Amazon price tracker), Honey browser extension, and PriceGrabber. These tools help you find the lowest prices and track price history to identify the best time to buy.",
                    "relevance_score": 0.85
                },
                {
                    "title": "Consumer Product Reviews: How to Spot Fake Reviews",
                    "url": "https://www.ftc.gov/consumer-advice/fake-reviews",
                    "snippet": "Warning signs of fake reviews: excessive 5-star ratings, generic language, posted on same day, overly promotional tone. Trust verified purchase reviews and look for detailed, balanced feedback. Check multiple sources including Reddit and specialized forums.",
                    "relevance_score": 0.83
                },
                {
                    "title": "E-commerce Trends 2024: What Consumers Are Buying",
                    "url": "https://www.shopify.com/blog/ecommerce-trends-2024",
                    "snippet": "Top trending product categories: smart home devices (up 35%), sustainable products (up 28%), health and wellness (up 25%), and home office equipment (stable). Average online order value increased to $85, up from $78 in 2023.",
                    "relevance_score": 0.80
                }
            ]
        
        # Return requested number of results
        return mock_data[:n_results]
    
    def _generate_answer(
        self,
        query: str,
        search_results: List[Dict[str, Any]]
    ) -> str:
        """
        Generate answer summary using LLM or fallback.
        
        Args:
            query: Original user query
            search_results: List of search results
        
        Returns:
            Answer summary string
        """
        if not search_results:
            return "No search results found for your query. Please try a different search term."
        
        if self.llm_available and self.llm:
            try:
                return self._generate_llm_answer(query, search_results)
            except Exception as e:
                logger.warning(f"LLM answer generation failed: {e}. Using fallback.")
                return self._generate_fallback_answer(query, search_results)
        else:
            return self._generate_fallback_answer(query, search_results)
    
    def _generate_llm_answer(
        self,
        query: str,
        search_results: List[Dict[str, Any]]
    ) -> str:
        """
        Generate answer using LLM.
        
        Args:
            query: User query
            search_results: Search results
        
        Returns:
            LLM-generated answer
        """
        # Build context from search results
        context = self._build_search_context(search_results)
        
        # Create messages for LangChain
        system_message = SystemMessage(content="""You are a helpful product research assistant.
Your task is to synthesize web search results into a clear, concise, and informative answer.
Focus on key information like prices, features, comparisons, and recommendations.
Be conversational but professional. Cite sources when mentioning specific information.""")
        
        human_message = HumanMessage(content=f"""User Query: {query}

Web Search Results:
{context}

Please provide a comprehensive answer to the user's query based on these search results.
Include relevant prices, product names, and key insights. Keep the response concise but informative.""")
        
        # Generate response using LangChain
        response = self.llm.invoke([system_message, human_message])
        # Handle both string and list responses from Azure OpenAI
        content = response.content if isinstance(response.content, str) else str(response.content[0]) if response.content else ""
        return content.strip()
    
    def _generate_fallback_answer(
        self,
        query: str,
        search_results: List[Dict[str, Any]]
    ) -> str:
        """
        Generate basic answer without LLM.
        
        Args:
            query: User query
            search_results: Search results
        
        Returns:
            Basic formatted answer
        """
        answer_parts = []
        
        # Add intro
        answer_parts.append(f"Found {len(search_results)} web results for your query.")
        
        # Add top results summary
        if search_results:
            answer_parts.append("\nTop results:")
            for i, result in enumerate(search_results[:3], 1):
                title = result.get("title", "Unknown")
                snippet = result.get("snippet", "")[:150]
                answer_parts.append(f"\n{i}. {title}")
                if snippet:
                    answer_parts.append(f"   {snippet}...")
        
        return "\n".join(answer_parts)
    
    def _build_search_context(self, search_results: List[Dict[str, Any]]) -> str:
        """
        Build formatted search context for LLM.
        
        Args:
            search_results: List of search results
        
        Returns:
            Formatted search context string
        """
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            title = result.get("title", "Unknown")
            url = result.get("url", "")
            snippet = result.get("snippet", "")
            score = result.get("relevance_score", 0.0)
            
            context = f"""
Result {i} (Relevance: {score:.2f}):
Title: {title}
URL: {url}
Content: {snippet}
"""
            context_parts.append(context.strip())
        
        return "\n\n".join(context_parts)
    
    def _create_error_response(
        self,
        query: str,
        error_message: str
    ) -> Dict[str, Any]:
        """
        Create error response structure.
        
        Args:
            query: Original query
            error_message: Error message
        
        Returns:
            Error response dictionary
        """
        return {
            "answer": f"An error occurred while searching: {error_message}",
            "results": [],
            "sources": [],
            "metadata": {
                "query": query,
                "api_used": "error",
                "n_results_requested": self.default_n_results,
                "n_results_returned": 0,
                "timestamp": datetime.utcnow().isoformat(),
                "llm_used": False,
                "error": error_message
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the web search tool.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "api_available": self.api_available,
            "api_type": "tavily" if self.api_available else "mock",
            "llm_available": self.llm_available,
            "default_n_results": self.default_n_results
        }


