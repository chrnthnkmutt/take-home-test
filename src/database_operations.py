"""
Database operations for the AI Product Research Assistant.

This module provides functions for creating, reading, updating, and deleting
queries and feedback in the database.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import uuid

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from src.database import get_db_context, init_db
from src.models import Query, Feedback


def save_query(
    query_text: str,
    tools_used: List[str],
    result: Dict[str, Any],
    response_time_ms: float
) -> Optional[str]:
    """
    Save a query and its results to the database.
    
    Args:
        query_text (str): The original query text from the user
        tools_used (List[str]): List of tools used to process the query
        result (Dict[str, Any]): The query result as a dictionary
        response_time_ms (float): Time taken to process the query in milliseconds
        
    Returns:
        Optional[str]: The UUID of the saved query as a string, or None if save failed
        
    Raises:
        SQLAlchemyError: If there's a database error during save
        
    Example:
        >>> query_id = save_query(
        ...     "What are the best laptops?",
        ...     ["search_products", "filter_products"],
        ...     {"products": [...]},
        ...     150.5
        ... )
    """
    try:
        with get_db_context() as db:
            # Create new query object
            new_query = Query(
                id=uuid.uuid4(),
                query_text=query_text,
                timestamp=datetime.utcnow(),
                tools_used=json.dumps(tools_used),
                result=json.dumps(result),
                response_time_ms=response_time_ms
            )
            
            # Add to database
            db.add(new_query)
            db.commit()
            db.refresh(new_query)
            
            return str(new_query.id)
            
    except SQLAlchemyError as e:
        print(f"Error saving query to database: {e}")
        return None


def get_all_queries() -> List[Dict[str, Any]]:
    """
    Retrieve all queries from the database.
    
    Returns:
        List[Dict[str, Any]]: List of all queries as dictionaries, ordered by timestamp (newest first)
        
    Raises:
        SQLAlchemyError: If there's a database error during retrieval
        
    Example:
        >>> queries = get_all_queries()
        >>> for query in queries:
        ...     print(query['query_text'])
    """
    try:
        with get_db_context() as db:
            # Query all records ordered by timestamp descending
            queries = db.query(Query).order_by(Query.timestamp.desc()).all()
            
            # Convert to dictionaries
            return [query.to_dict() for query in queries]
            
    except SQLAlchemyError as e:
        print(f"Error retrieving queries from database: {e}")
        return []


def save_feedback(
    query_id: str,
    rating: int,
    comment: Optional[str] = None
) -> Optional[str]:
    """
    Save user feedback for a specific query.
    
    Args:
        query_id (str): UUID of the query to provide feedback for
        rating (int): User rating from 1 to 5
        comment (Optional[str]): Optional user comment about the query result
        
    Returns:
        Optional[str]: The UUID of the saved feedback as a string, or None if save failed
        
    Raises:
        ValueError: If rating is not between 1 and 5
        SQLAlchemyError: If there's a database error during save
        
    Example:
        >>> feedback_id = save_feedback(
        ...     "123e4567-e89b-12d3-a456-426614174000",
        ...     5,
        ...     "Very helpful results!"
        ... )
    """
    # Validate rating
    if not isinstance(rating, int) or rating < 1 or rating > 5:
        raise ValueError("Rating must be an integer between 1 and 5")
    
    try:
        with get_db_context() as db:
            # Verify query exists
            query = db.query(Query).filter(Query.id == uuid.UUID(query_id)).first()
            if not query:
                print(f"Query with id {query_id} not found")
                return None
            
            # Create new feedback object
            new_feedback = Feedback(
                id=uuid.uuid4(),
                query_id=uuid.UUID(query_id),
                rating=rating,
                comment=comment,
                timestamp=datetime.utcnow()
            )
            
            # Add to database
            db.add(new_feedback)
            db.commit()
            db.refresh(new_feedback)
            
            return str(new_feedback.id)
            
    except ValueError as e:
        print(f"Invalid query_id format: {e}")
        return None
    except SQLAlchemyError as e:
        print(f"Error saving feedback to database: {e}")
        return None


def get_query_by_id(query_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a specific query by its ID.
    
    Args:
        query_id (str): UUID of the query to retrieve
        
    Returns:
        Optional[Dict[str, Any]]: Query as a dictionary, or None if not found
        
    Raises:
        SQLAlchemyError: If there's a database error during retrieval
        
    Example:
        >>> query = get_query_by_id("123e4567-e89b-12d3-a456-426614174000")
        >>> if query:
        ...     print(query['query_text'])
    """
    try:
        with get_db_context() as db:
            # Query by ID
            query = db.query(Query).filter(Query.id == uuid.UUID(query_id)).first()
            
            if query:
                return query.to_dict()
            else:
                print(f"Query with id {query_id} not found")
                return None
                
    except ValueError as e:
        print(f"Invalid query_id format: {e}")
        return None
    except SQLAlchemyError as e:
        print(f"Error retrieving query from database: {e}")
        return None


def get_feedback_by_query_id(query_id: str) -> List[Dict[str, Any]]:
    """
    Retrieve all feedback for a specific query.
    
    Args:
        query_id (str): UUID of the query to get feedback for
        
    Returns:
        List[Dict[str, Any]]: List of feedback entries as dictionaries
        
    Raises:
        SQLAlchemyError: If there's a database error during retrieval
        
    Example:
        >>> feedbacks = get_feedback_by_query_id("123e4567-e89b-12d3-a456-426614174000")
        >>> for feedback in feedbacks:
        ...     print(f"Rating: {feedback['rating']}")
    """
    try:
        with get_db_context() as db:
            # Query feedback by query_id
            feedbacks = db.query(Feedback).filter(
                Feedback.query_id == uuid.UUID(query_id)
            ).order_by(Feedback.timestamp.desc()).all()
            
            return [feedback.to_dict() for feedback in feedbacks]
            
    except ValueError as e:
        print(f"Invalid query_id format: {e}")
        return []
    except SQLAlchemyError as e:
        print(f"Error retrieving feedback from database: {e}")
        return []


def delete_query(query_id: str) -> bool:
    """
    Delete a query and all its associated feedback from the database.
    
    Args:
        query_id (str): UUID of the query to delete
        
    Returns:
        bool: True if deletion was successful, False otherwise
        
    Raises:
        SQLAlchemyError: If there's a database error during deletion
        
    Example:
        >>> success = delete_query("123e4567-e89b-12d3-a456-426614174000")
        >>> if success:
        ...     print("Query deleted successfully")
    """
    try:
        with get_db_context() as db:
            # Find and delete query (cascade will delete associated feedback)
            query = db.query(Query).filter(Query.id == uuid.UUID(query_id)).first()
            
            if query:
                db.delete(query)
                db.commit()
                return True
            else:
                print(f"Query with id {query_id} not found")
                return False
                
    except ValueError as e:
        print(f"Invalid query_id format: {e}")
        return False
    except SQLAlchemyError as e:
        print(f"Error deleting query from database: {e}")
        return False


# Initialize database when module is imported
init_db()


