"""Pydantic schemas for Article requests and responses."""

from datetime import datetime
from typing import Optional

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
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
