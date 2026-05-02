"""
SQLAlchemy models for the AI Product Research Assistant.

This module defines the database models for storing queries and user feedback.
"""

from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.types import TypeDecorator, CHAR
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import json

from src.database import Base


class GUID(TypeDecorator):
    """
    Platform-independent GUID type.
    
    Uses PostgreSQL's UUID type when available, otherwise uses
    CHAR(36) storing as stringified hex values.
    """
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(PG_UUID())
        else:
            return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return str(value)
        else:
            if not isinstance(value, uuid.UUID):
                return str(uuid.UUID(value))
            else:
                return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        else:
            if not isinstance(value, uuid.UUID):
                return uuid.UUID(value)
            return value


class Query(Base):
    """
    Model for storing user queries and their results.
    
    Attributes:
        id (UUID): Primary key, unique identifier for the query
        query_text (str): The original query text from the user
        timestamp (datetime): When the query was executed
        tools_used (str): JSON string of tools used to process the query
        result (str): JSON string of the query result
        response_time_ms (float): Time taken to process the query in milliseconds
        feedbacks (relationship): Related feedback entries for this query
    """
    __tablename__ = "queries"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4, nullable=False)
    query_text = Column(Text, nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    tools_used = Column(Text, nullable=False)  # JSON string
    result = Column(Text, nullable=False)  # JSON string
    response_time_ms = Column(Float, nullable=False)

    # Relationship to feedback
    feedbacks = relationship("Feedback", back_populates="query", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        """String representation of the Query object."""
        return f"<Query(id={self.id}, query_text='{self.query_text[:50]}...', timestamp={self.timestamp})>"

    def to_dict(self) -> dict:
        """
        Convert the Query object to a dictionary.
        
        Returns:
            dict: Dictionary representation of the query with parsed JSON fields
        """
        return {
            "id": str(self.id),
            "query_text": self.query_text,
            "timestamp": self.timestamp.isoformat(),
            "tools_used": json.loads(self.tools_used),
            "result": json.loads(self.result),
            "response_time_ms": self.response_time_ms,
            "feedbacks": [feedback.to_dict() for feedback in self.feedbacks]
        }


class Feedback(Base):
    """
    Model for storing user feedback on queries.
    
    Attributes:
        id (UUID): Primary key, unique identifier for the feedback
        query_id (UUID): Foreign key referencing the related query
        rating (int): User rating from 1 to 5
        comment (str): Optional user comment about the query result
        timestamp (datetime): When the feedback was submitted
        query (relationship): Related query object
    """
    __tablename__ = "feedbacks"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4, nullable=False)
    query_id = Column(GUID(), ForeignKey("queries.id", ondelete="CASCADE"), nullable=False, index=True)
    rating = Column(Integer, nullable=False)
    comment = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationship to query
    query = relationship("Query", back_populates="feedbacks")

    def __repr__(self) -> str:
        """String representation of the Feedback object."""
        return f"<Feedback(id={self.id}, query_id={self.query_id}, rating={self.rating})>"

    def to_dict(self) -> dict:
        """
        Convert the Feedback object to a dictionary.
        
        Returns:
            dict: Dictionary representation of the feedback
        """
        return {
            "id": str(self.id),
            "query_id": str(self.query_id),
            "rating": self.rating,
            "comment": self.comment,
            "timestamp": self.timestamp.isoformat()
        }


