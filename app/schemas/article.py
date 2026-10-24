"""Pydantic schemas for Article requests and responses."""

from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ArticleBase(BaseModel):
    """Base schema for Article metadata."""

    title: str = Field(..., max_length=255, description="Article title")
    content: str = Field(..., description="Full text or summary of the article")
    original_language: str = Field("vi", max_length=10, description="Language code")
    author: Optional[str] = Field(None, max_length=100, description="Author or contributor name")


class ArticleCreate(ArticleBase):
    """Schema for creating a new Article."""

    pass


class ArticleResponse(ArticleBase):
    """Schema for returning Article entity details."""

    id: int
    has_embedding: bool = Field(default=False, description="Whether vector embedding has been computed")
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


from app.schemas.video import VideoDetailResponse


class ArticleDetailResponse(ArticleResponse):
    """Schema for detailed Article information including attached videos."""

    videos: List[VideoDetailResponse] = Field(default_factory=list, description="Associated video records")

    model_config = ConfigDict(from_attributes=True)


class ArticleSearchRequest(BaseModel):
    """Request payload for semantic similarity search over articles."""

    query: str = Field(..., description="Search text or question to find relevant articles")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of articles to return")
    min_similarity: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional minimum similarity score threshold (0.0 to 1.0)",
    )


class ArticleSearchResultItem(ArticleBase):
    """Schema for an individual article similarity search match."""

    id: int
    similarity_score: float = Field(..., description="Cosine similarity score (0.0 to 1.0, higher is more relevant)")
    distance: float = Field(..., description="Cosine distance metric (0.0 to 2.0, lower is closer)")
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ArticleSearchResponse(BaseModel):
    """Response containing search results ordered by relevance score."""

    query: str
    total_results: int
    results: List[ArticleSearchResultItem] = Field(default_factory=list)


