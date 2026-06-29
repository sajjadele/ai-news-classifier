"""Data models for the AI News Classifier."""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class Article(BaseModel):
    """A fetched news article."""
    title: str
    content: str
    url: Optional[str] = None
    source: Optional[str] = None
    published_at: Optional[datetime] = None


class ClassificationResult(BaseModel):
    """Classification output for an article."""
    relevant: bool = Field(description="Whether the article is AI-related")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    reason: str = Field(description="Short technical justification")
    article_title: str = Field(description="Original article title")
    article_url: Optional[str] = None
    article_id: Optional[int] = None
    error: Optional[str] = None


class RunSummary(BaseModel):
    """Summary of a classification run."""
    total: int
    relevant: int
    not_relevant: int
    avg_confidence: float
    results: list[ClassificationResult]
