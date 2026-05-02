"""
Database configuration and session management for the AI Product Research Assistant.

This module provides SQLAlchemy setup, database initialization, and session management
for storing queries and feedback.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator
import os

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///queries.db")

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    echo=False  # Set to True for SQL query logging during development
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create declarative base for models
Base = declarative_base()


def init_db() -> None:
    """
    Initialize the database by creating all tables.
    
    This function should be called once when the application starts
    to ensure all required tables exist in the database.
    """
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Get a database session.
    
    This is a generator function that yields a database session and ensures
    it's properly closed after use. Typically used with dependency injection
    in FastAPI or similar frameworks.
    
    Yields:
        Session: SQLAlchemy database session
        
    Example:
        >>> for db in get_db():
        ...     # Use db session here
        ...     pass
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    Get a database session as a context manager.
    
    This context manager ensures the database session is properly closed
    even if an exception occurs.
    
    Yields:
        Session: SQLAlchemy database session
        
    Example:
        >>> with get_db_context() as db:
        ...     # Use db session here
        ...     pass
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def close_db() -> None:
    """
    Close all database connections and dispose of the engine.
    
    This function should be called when shutting down the application
    to ensure all database connections are properly closed.
    """
    engine.dispose()


