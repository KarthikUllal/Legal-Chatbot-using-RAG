# backend/app/news_schemas.py
"""
Pydantic schemas for legal news
"""
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List, Dict, Any
from datetime import datetime


# Basic legal news/article structure with required and optional fields
class LegalArticleBase(BaseModel):
    """Base schema for legal article"""
    id: str
    title: str
    description: Optional[str] = None
    summary: Optional[str] = None
    content: Optional[str] = None
    link: str
    published: Optional[str] = None
    source: str
    source_id: str
    category: str
    image: Optional[str] = None
    read_time: int = Field(default=1, ge=1)


# Extended article response with authors, keywords for frontend display 
class LegalArticleResponse(LegalArticleBase):
    """Response schema for legal article"""
    authors: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    images: Optional[List[str]] = None
    top_image: Optional[str] = None
    # Remove published_parsed or make it optional string
    published_parsed: Optional[Any] = None  # Allow any type or remove it

    class Config:
        arbitrary_types_allowed = True  # Allow non-standard types


# Legal news source definition (LiveLaw, BarAndBench) for scraping config
class NewsSource(BaseModel):
    """Schema for news source"""
    id: str
    name: str
    url: str
    type: str
    enabled: bool = True


# Legal news categories (Supreme Court, Criminal Law) for organization
# class NewsCategory(BaseModel):
#     """Schema for news category"""
#     id: str
#     name: str
#     description: Optional[str] = None
#     icon: Optional[str] = None

# # Request schema for searching legal news with query/filters
# class NewsSearchRequest(BaseModel):
#     """Schema for news search request"""
#     query: str
#     category: Optional[str] = None
#     source: Optional[str] = None
#     limit: int = Field(default=20, ge=1, le=100)


# # Advanced filtering with multiple categories, sources, date ranges
# class NewsFilterRequest(BaseModel):
#     """Schema for news filter request"""
#     categories: Optional[List[str]] = None
#     sources: Optional[List[str]] = None
#     date_from: Optional[str] = None
#     date_to: Optional[str] = None
#     limit: int = Field(default=20, ge=1, le=100)


# Response for news update operations showing status and results
class NewsUpdateResponse(BaseModel):
    """Schema for news update response"""
    status: str = "success"
    total_articles: int
    articles: List[Dict]  # Use Dict instead of LegalArticleResponse for flexibility
    last_updated: str
    sources: List[str]