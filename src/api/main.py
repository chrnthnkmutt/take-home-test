"""
FastAPI application for the AI Product Research Assistant.

This module provides REST API endpoints for querying the product research agent,
retrieving query history, submitting feedback, and checking system health.
"""

import logging
import asyncio
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.agent.agent import ProductResearchAgent
from src.pipeline.ingestion import ingest_products
from src.database import init_db
from src.database_operations import save_query, get_all_queries, save_feedback
from src.api.models import (
    QueryRequest,
    QueryResponse,
    FeedbackRequest,
    FeedbackResponse,
    QueryHistoryResponse,
    QueryHistoryItem,
    HealthResponse,
    ComponentHealth,
    ErrorResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API version
API_VERSION = "1.0.0"

# Initialize FastAPI app
app = FastAPI(
    title="AI Product Research Assistant API",
    description="REST API for the AI Product Research Assistant with intelligent tool routing",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance (initialized on startup)
agent: Optional[ProductResearchAgent] = None


@app.on_event("startup")
async def startup_event():
    """
    Initialize the application on startup.
    
    This function:
    - Initializes the database
    - Creates the ProductResearchAgent instance
    - Logs startup information
    """
    logger.info("Starting AI Product Research Assistant API...")
    
    try:
        # Initialize database
        init_db()
        logger.info("Database initialized successfully")
        
        # Initialize agent
        global agent
        agent = ProductResearchAgent()
        logger.info("ProductResearchAgent initialized successfully")
        
        # Log agent stats
        stats = agent.get_stats()
        logger.info(f"Agent routing mode: {stats['agent']['routing_mode']}")
        logger.info(f"LLM available: {stats['agent']['llm_available']}")
        
        logger.info(f"API v{API_VERSION} started successfully")

        # If the vector store has no documents, start ingestion in background
        try:
            if agent and agent.catalog_tool:
                catalog_stats = agent.catalog_tool.get_stats()
                total_docs = catalog_stats.get("total_documents", 0)
                if total_docs == 0:
                    logger.info("Vector store empty — scheduling background ingestion task")

                    async def _run_ingestion():
                        try:
                            # Run the synchronous ingestion function in a thread to avoid blocking
                            result = await asyncio.to_thread(ingest_products)
                            logger.info(f"Background ingestion finished: {result}")
                        except Exception as ie:
                            logger.error(f"Background ingestion failed: {ie}")

                    asyncio.create_task(_run_ingestion())

        except Exception as e:
            logger.warning(f"Failed to check/schedule ingestion: {e}")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """
    Clean up resources on shutdown.
    """
    logger.info("Shutting down AI Product Research Assistant API...")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unhandled errors.
    
    Args:
        request: The request that caused the error
        exc: The exception that was raised
        
    Returns:
        JSONResponse with error details
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    error_response = ErrorResponse(
        error="InternalServerError",
        message=str(exc),
        timestamp=datetime.utcnow().isoformat()
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.dict()
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Process a product research query",
    description="Submit a natural language query to the AI agent and receive intelligent responses"
)
async def process_query(request: QueryRequest) -> QueryResponse:
    """
    Process a user query using the ProductResearchAgent.
    
    This endpoint:
    1. Validates the query
    2. Processes it using the agent (which selects and executes appropriate tools)
    3. Saves the query and results to the database
    4. Returns the agent's response with a query_id
    
    Args:
        request: QueryRequest containing the user's query
        
    Returns:
        QueryResponse with the agent's answer and metadata
        
    Raises:
        HTTPException: If query processing fails
    """
    logger.info(f"Received query: {request.query}")
    
    try:
        # Check if agent is initialized
        if agent is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Agent not initialized. Please try again later."
            )
        
        # Process query with agent
        result = agent.process_query(request.query)
        
        # Check if processing was successful
        if not result.get("metadata", {}).get("success", False):
            error_msg = result.get("metadata", {}).get("error", "Unknown error")
            logger.error(f"Query processing failed: {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Query processing failed: {error_msg}"
            )
        
        # Save query to database
        query_id = save_query(
            query_text=request.query,
            tools_used=result["tools_used"],
            result=result,
            response_time_ms=result["metadata"]["execution_time_ms"]
        )
        
        if not query_id:
            logger.warning("Failed to save query to database, but returning result anyway")
            query_id = "unsaved"
        
        logger.info(f"Query processed successfully. Query ID: {query_id}")
        
        # Build response
        response = QueryResponse(
            query_id=query_id,
            query=result["query"],
            reasoning=result["reasoning"],
            tools_used=result["tools_used"],
            results=result["results"],
            final_answer=result["final_answer"],
            metadata=result["metadata"]
        )
        
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your query: {str(e)}"
        )


@app.get(
    "/queries",
    response_model=QueryHistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve query history",
    description="Get a list of all previous queries with optional pagination"
)
async def get_queries(
    limit: Optional[int] = Query(None, ge=1, le=100, description="Maximum number of queries to return"),
    offset: Optional[int] = Query(None, ge=0, description="Number of queries to skip")
) -> QueryHistoryResponse:
    """
    Retrieve all queries from the database with optional pagination.
    
    Args:
        limit: Maximum number of queries to return (1-100)
        offset: Number of queries to skip (for pagination)
        
    Returns:
        QueryHistoryResponse with list of queries and metadata
        
    Raises:
        HTTPException: If retrieval fails
    """
    logger.info(f"Retrieving queries (limit={limit}, offset={offset})")
    
    try:
        # Get all queries from database
        all_queries = get_all_queries()
        total = len(all_queries)
        
        # Apply pagination if specified
        if offset is not None:
            all_queries = all_queries[offset:]
        
        if limit is not None:
            all_queries = all_queries[:limit]
        
        # Convert to QueryHistoryItem models
        query_items = [
            QueryHistoryItem(
                id=q["id"],
                query_text=q["query_text"],
                timestamp=q["timestamp"],
                tools_used=q["tools_used"],
                response_time_ms=q["response_time_ms"],
                feedbacks=q["feedbacks"]
            )
            for q in all_queries
        ]
        
        logger.info(f"Retrieved {len(query_items)} queries (total: {total})")
        
        response = QueryHistoryResponse(
            queries=query_items,
            total=total,
            limit=limit,
            offset=offset
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving queries: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while retrieving queries: {str(e)}"
        )


@app.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit user feedback",
    description="Submit feedback (rating and optional comment) for a specific query"
)
async def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """
    Submit user feedback for a specific query.
    
    Args:
        request: FeedbackRequest containing query_id, rating, and optional comment
        
    Returns:
        FeedbackResponse with confirmation and feedback details
        
    Raises:
        HTTPException: If feedback submission fails or query not found
    """
    logger.info(f"Received feedback for query {request.query_id}: rating={request.rating}")
    
    try:
        # Save feedback to database
        feedback_id = save_feedback(
            query_id=request.query_id,
            rating=request.rating,
            comment=request.comment
        )
        
        if not feedback_id:
            logger.error(f"Failed to save feedback for query {request.query_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Query with id {request.query_id} not found"
            )
        
        logger.info(f"Feedback saved successfully. Feedback ID: {feedback_id}")
        
        # Build response
        response = FeedbackResponse(
            feedback_id=feedback_id,
            query_id=request.query_id,
            rating=request.rating,
            comment=request.comment,
            timestamp=datetime.utcnow().isoformat(),
            message="Feedback submitted successfully"
        )
        
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while submitting feedback: {str(e)}"
        )


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Check the health status of the API and its components"
)
async def health_check() -> HealthResponse:
    """
    Check the health status of the API and its components.
    
    This endpoint checks:
    - Agent availability and LLM status
    - Vector database connection
    - Overall system health
    
    Returns:
        HealthResponse with status of all components
    """
    logger.info("Health check requested")
    
    components = {}
    overall_status = "healthy"
    
    try:
        # Check agent
        if agent is None:
            components["agent"] = ComponentHealth(
                status="unhealthy",
                message="Agent not initialized"
            )
            overall_status = "unhealthy"
        else:
            stats = agent.get_stats()
            agent_status = "healthy" if stats["agent"]["llm_available"] else "degraded"
            components["agent"] = ComponentHealth(
                status=agent_status,
                message=f"Agent running in {stats['agent']['routing_mode']} mode"
            )
            if agent_status == "degraded" and overall_status == "healthy":
                overall_status = "degraded"
        
        # Check vector database (via catalog tool)
        try:
            if agent and agent.catalog_tool:
                catalog_stats = agent.catalog_tool.get_stats()
                if catalog_stats.get("vector_store_initialized", False):
                    components["vector_database"] = ComponentHealth(
                        status="healthy",
                        message=f"Vector database connected with {catalog_stats.get('total_documents', 0)} documents"
                    )
                else:
                    components["vector_database"] = ComponentHealth(
                        status="unhealthy",
                        message="Vector database not initialized"
                    )
                    overall_status = "unhealthy"
            else:
                components["vector_database"] = ComponentHealth(
                    status="unknown",
                    message="Cannot check vector database status"
                )
                if overall_status == "healthy":
                    overall_status = "degraded"
        except Exception as e:
            logger.warning(f"Error checking vector database: {e}")
            components["vector_database"] = ComponentHealth(
                status="unhealthy",
                message=f"Error checking vector database: {str(e)}"
            )
            overall_status = "unhealthy"
        
        # Check LLM availability
        if agent:
            llm_status = "healthy" if agent.llm_available else "degraded"
            components["llm"] = ComponentHealth(
                status=llm_status,
                message="LLM available for routing and aggregation" if agent.llm_available else "LLM not available, using fallback methods"
            )
        else:
            components["llm"] = ComponentHealth(
                status="unknown",
                message="Cannot check LLM status"
            )
        
        # Check database
        try:
            # Try to get queries to verify database connection
            get_all_queries()
            components["database"] = ComponentHealth(
                status="healthy",
                message="Database connection successful"
            )
        except Exception as e:
            logger.warning(f"Error checking database: {e}")
            components["database"] = ComponentHealth(
                status="unhealthy",
                message=f"Database connection failed: {str(e)}"
            )
            overall_status = "unhealthy"
        
        logger.info(f"Health check complete. Overall status: {overall_status}")
        
        response = HealthResponse(
            status=overall_status,
            timestamp=datetime.utcnow().isoformat(),
            components=components,
            version=API_VERSION
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error during health check: {e}", exc_info=True)
        
        # Return unhealthy status
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow().isoformat(),
            components={
                "system": ComponentHealth(
                    status="unhealthy",
                    message=f"Health check failed: {str(e)}"
                )
            },
            version=API_VERSION
        )


# Root endpoint
@app.get(
    "/",
    summary="API root",
    description="Get basic information about the API"
)
async def root():
    """
    Root endpoint providing basic API information.
    
    Returns:
        Dictionary with API name, version, and documentation links
    """
    return {
        "name": "AI Product Research Assistant API",
        "version": API_VERSION,
        "description": "REST API for intelligent product research with multi-tool routing",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health"
    }


