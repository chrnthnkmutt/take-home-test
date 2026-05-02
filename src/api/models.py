"""
Pydantic models for API request/response validation.

This module defines the data models used for validating requests and responses
in the FastAPI application.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime


class QueryRequest(BaseModel):
    """
    Request model for POST /query endpoint.
    
    Attributes:
        query: The user's natural language query
    """
    query: str = Field(
        ...,
        description="Natural language query for the product research assistant",
        min_length=1,
        max_length=1000,
        example="What wireless headphones do we have under $200?"
    )
    
    @validator('query')
    def query_not_empty(cls, v):
        """Validate that query is not just whitespace."""
        if not v or not v.strip():
            raise ValueError('Query cannot be empty or whitespace only')
        return v.strip()


class QueryResponse(BaseModel):
    """
    Response model for POST /query endpoint.
    
    Attributes:
        query_id: Unique identifier for the saved query
        query: Original user query
        reasoning: Explanation of tool selection
        tools_used: List of tools that were used
        results: Dictionary of results from each tool
        final_answer: Aggregated answer combining all tool results
        metadata: Execution metadata
    """
    query_id: str = Field(
        ...,
        description="Unique identifier for the query",
        example="123e4567-e89b-12d3-a456-426614174000"
    )
    query: str = Field(
        ...,
        description="Original user query",
        example="What wireless headphones do we have?"
    )
    reasoning: str = Field(
        ...,
        description="Explanation of why these tools were selected",
        example="Query asks about internal product catalog"
    )
    tools_used: List[str] = Field(
        ...,
        description="List of tools used to process the query",
        example=["ProductCatalogRAG"]
    )
    results: Dict[str, Any] = Field(
        ...,
        description="Detailed results from each tool"
    )
    final_answer: str = Field(
        ...,
        description="Aggregated answer combining all tool results"
    )
    metadata: Dict[str, Any] = Field(
        ...,
        description="Execution metadata including timestamp and execution time"
    )


class FeedbackRequest(BaseModel):
    """
    Request model for POST /feedback endpoint.
    
    Attributes:
        query_id: UUID of the query to provide feedback for
        rating: User rating from 1 to 5
        comment: Optional user comment
    """
    query_id: str = Field(
        ...,
        description="UUID of the query to provide feedback for",
        example="123e4567-e89b-12d3-a456-426614174000"
    )
    rating: int = Field(
        ...,
        description="User rating from 1 to 5",
        ge=1,
        le=5,
        example=5
    )
    comment: Optional[str] = Field(
        None,
        description="Optional user comment about the query result",
        max_length=1000,
        example="Very helpful results!"
    )
    
    @validator('rating')
    def validate_rating(cls, v):
        """Validate that rating is between 1 and 5."""
        if v < 1 or v > 5:
            raise ValueError('Rating must be between 1 and 5')
        return v


class FeedbackResponse(BaseModel):
    """
    Response model for POST /feedback endpoint.
    
    Attributes:
        feedback_id: Unique identifier for the saved feedback
        query_id: UUID of the query the feedback is for
        rating: User rating
        comment: User comment (if provided)
        timestamp: When the feedback was submitted
        message: Confirmation message
    """
    feedback_id: str = Field(
        ...,
        description="Unique identifier for the feedback",
        example="123e4567-e89b-12d3-a456-426614174000"
    )
    query_id: str = Field(
        ...,
        description="UUID of the query",
        example="123e4567-e89b-12d3-a456-426614174000"
    )
    rating: int = Field(
        ...,
        description="User rating",
        example=5
    )
    comment: Optional[str] = Field(
        None,
        description="User comment",
        example="Very helpful results!"
    )
    timestamp: str = Field(
        ...,
        description="ISO 8601 timestamp of when feedback was submitted",
        example="2024-01-01T12:00:00.000Z"
    )
    message: str = Field(
        ...,
        description="Confirmation message",
        example="Feedback submitted successfully"
    )


class QueryHistoryItem(BaseModel):
    """
    Model for a single query in the history list.
    
    Attributes:
        id: Query UUID
        query_text: Original query text
        timestamp: When the query was executed
        tools_used: List of tools used
        response_time_ms: Response time in milliseconds
        feedbacks: List of feedback entries for this query
    """
    id: str
    query_text: str
    timestamp: str
    tools_used: List[str]
    response_time_ms: float
    feedbacks: List[Dict[str, Any]]


class QueryHistoryResponse(BaseModel):
    """
    Response model for GET /queries endpoint.
    
    Attributes:
        queries: List of query history items
        total: Total number of queries
        limit: Number of queries returned (if pagination used)
        offset: Offset used (if pagination used)
    """
    queries: List[QueryHistoryItem] = Field(
        ...,
        description="List of queries with their metadata"
    )
    total: int = Field(
        ...,
        description="Total number of queries in the database",
        example=42
    )
    limit: Optional[int] = Field(
        None,
        description="Number of queries returned (if pagination used)",
        example=10
    )
    offset: Optional[int] = Field(
        None,
        description="Offset used (if pagination used)",
        example=0
    )


class ComponentHealth(BaseModel):
    """
    Model for individual component health status.
    
    Attributes:
        status: Component status (healthy, degraded, unhealthy)
        message: Additional information about the component
    """
    status: str = Field(
        ...,
        description="Component status",
        example="healthy"
    )
    message: Optional[str] = Field(
        None,
        description="Additional information about the component",
        example="Vector database connected successfully"
    )


class HealthResponse(BaseModel):
    """
    Response model for GET /health endpoint.
    
    Attributes:
        status: Overall system status
        timestamp: Current timestamp
        components: Health status of individual components
        version: API version
    """
    status: str = Field(
        ...,
        description="Overall system status (healthy, degraded, unhealthy)",
        example="healthy"
    )
    timestamp: str = Field(
        ...,
        description="Current ISO 8601 timestamp",
        example="2024-01-01T12:00:00.000Z"
    )
    components: Dict[str, ComponentHealth] = Field(
        ...,
        description="Health status of individual components"
    )
    version: str = Field(
        ...,
        description="API version",
        example="1.0.0"
    )


class ErrorResponse(BaseModel):
    """
    Standard error response model.
    
    Attributes:
        error: Error type or category
        message: Detailed error message
        timestamp: When the error occurred
    """
    error: str = Field(
        ...,
        description="Error type or category",
        example="ValidationError"
    )
    message: str = Field(
        ...,
        description="Detailed error message",
        example="Query cannot be empty"
    )
    timestamp: str = Field(
        ...,
        description="ISO 8601 timestamp of when error occurred",
        example="2024-01-01T12:00:00.000Z"
    )


