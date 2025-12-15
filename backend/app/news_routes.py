# Add to backend/app/main.py or create backend/app/news_routes.py

from chromadb import logger
from .news_service import news_service
from .news_schemas import (
    LegalArticleResponse,  NewsUpdateResponse
)
import logging
from fastapi import FastAPI,  HTTPException, Query
from datetime import datetime
import re
from typing import Optional, List
from datetime import datetime

# Create router if separate file
from fastapi import APIRouter
news_router = APIRouter(prefix="/news", tags=["legal-news"])

logger = logging.getLogger(__name__)

news_router = APIRouter(prefix="/news", tags=["legal-news"])

@news_router.get("/")
async def get_latest_legal_news(
    limit: int = Query(20, ge=1, le=100),
    category: Optional[str] = None,
    source: Optional[str] = None
):
    """Get latest legal news from RSS feeds"""
    try:
        # Fetch all news
        articles = await news_service.fetch_all_news(limit_per_source=5)
        
        # Apply filters if provided
        filtered_articles = articles
        
        if category:
            category_name = news_service.categories.get(category)
            if category_name:
                filtered_articles = [a for a in filtered_articles if a.get("category") == category_name]
        
        if source:
            filtered_articles = [a for a in filtered_articles if a.get("source_id") == source]
        
        # Apply limit
        filtered_articles = filtered_articles[:limit]
        
        # Clean up articles (remove internal fields)
        cleaned_articles = []
        for article in filtered_articles:
            # Create a clean copy without internal fields
            clean_article = {
                "id": article.get("id", ""),
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "summary": article.get("summary", ""),
                "content": article.get("content", ""),
                "link": article.get("link", ""),
                "published": article.get("published", ""),
                "source": article.get("source", ""),
                "source_id": article.get("source_id", ""),
                "category": article.get("category", ""),
                "image": article.get("image", ""),
                "read_time": article.get("read_time", 1)
            }
            # Remove empty values
            clean_article = {k: v for k, v in clean_article.items() if v is not None and v != ""}
            cleaned_articles.append(clean_article)
        
        return {
            "status": "success",
            "total_articles": len(articles),
            "articles": cleaned_articles,
            "filters_applied": {
                "category": category,
                "source": source,
                "limit": limit
            },
            "last_updated": datetime.now().isoformat(),
            "sources": list(news_service.sources.keys())
        }
        
    except Exception as e:
        logger.error(f"Error fetching news: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to fetch legal news: {str(e)}"
        )

@news_router.get("/categories")
async def get_news_categories():
    """Get available news categories"""
    try:
        return {
            "status": "success",
            "categories": news_service.categories,
            "count": len(news_service.categories)
        }
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        raise HTTPException(status_code=500, detail="Failed to get categories")

@news_router.get("/sources")
async def get_news_sources():
    """Get available news sources"""
    try:
        # Prepare source info for response
        sources_info = {}
        for source_id, source in news_service.sources.items():
            sources_info[source_id] = {
                "name": source["name"],
                "rss_url": source.get("rss_url", source.get("url", "")),
                "website_url": source.get("website_url", ""),
                "type": source["type"]
            }
        
        return {
            "status": "success", 
            "sources": sources_info,
            "count": len(sources_info)
        }
    except Exception as e:
        logger.error(f"Error getting sources: {e}")
        raise HTTPException(status_code=500, detail="Failed to get sources")

@news_router.get("/category/{category_name}")
async def get_news_by_category(
    category_name: str,
    limit: int = Query(20, ge=1, le=100)
):
    """Get news by category name"""
    try:
        # Map category name to internal ID
        category_id = None
        for cat_id, cat_name in news_service.categories.items():
            if cat_name.lower() == category_name.lower() or cat_id.lower() == category_name.lower():
                category_id = cat_id
                break
        
        if not category_id:
            raise HTTPException(status_code=404, detail=f"Category '{category_name}' not found")
        
        articles = news_service.get_legal_updates_by_category(news_service.categories[category_id])
        
        # Clean articles
        cleaned_articles = []
        for article in articles[:limit]:
            clean_article = {
                "id": article.get("id", ""),
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "link": article.get("link", ""),
                "published": article.get("published", ""),
                "source": article.get("source", ""),
                "category": article.get("category", ""),
                "read_time": article.get("read_time", 1)
            }
            cleaned_articles.append(clean_article)
        
        return {
            "status": "success",
            "category": news_service.categories[category_id],
            "articles": cleaned_articles,
            "count": len(cleaned_articles)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching category news: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch category news")

@news_router.get("/search")
async def search_legal_news(
    q: str = Query(..., min_length=2, description="Search query"),
    category: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100)
):
    """Search legal news"""
    try:
        results = news_service.search_legal_news(q)
        
        # Apply category filter if provided
        if category:
            category_name = news_service.categories.get(category)
            if category_name:
                results = [r for r in results if r.get("category") == category_name]
        
        # Clean results
        cleaned_results = []
        for article in results[:limit]:
            clean_article = {
                "id": article.get("id", ""),
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "link": article.get("link", ""),
                "published": article.get("published", ""),
                "source": article.get("source", ""),
                "category": article.get("category", ""),
                "relevance_score": 100 - len(cleaned_results)  # Simple scoring
            }
            cleaned_results.append(clean_article)
        
        return {
            "status": "success",
            "query": q,
            "results": cleaned_results,
            "total": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error searching news: {e}")
        raise HTTPException(status_code=500, detail="Search failed")

@news_router.get("/trending")
async def get_trending_topics(limit: int = Query(10, ge=1, le=20)):
    """Get trending legal topics"""
    try:
        articles = await news_service.fetch_all_news(limit_per_source=3)
        
        # Simple trending detection
        keyword_counts = {}
        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')}".lower()
            
            # Extract words
            words = re.findall(r'\b[a-z]{4,}\b', text)
            
            # Filter common words
            common_words = {
                "that", "with", "this", "from", "have", "what", "when", "were",
                "their", "there", "which", "would", "about", "could", "should",
                "legal", "court", "judgment", "section", "india", "indian"
            }
            
            for word in words:
                if word not in common_words:
                    keyword_counts[word] = keyword_counts.get(word, 0) + 1
        
        # Get top keywords
        sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
        trending = []
        
        for keyword, count in sorted_keywords[:limit]:
            trending.append({
                "topic": keyword.title(),
                "count": count,
                "trend": "rising" if count > 3 else "stable"
            })
        
        return {
            "status": "success",
            "trending_topics": trending,
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error detecting trends: {e}")
        return {
            "status": "error",
            "trending_topics": [],
            "error": str(e)
        }

@news_router.get("/refresh")
async def refresh_news_cache():
    """Force refresh news cache"""
    try:
        # Clear cache
        news_service.cache = {}
        
        # Fetch fresh news
        articles = await news_service.fetch_all_news(limit_per_source=5)
        
        return {
            "status": "success",
            "message": "News cache refreshed successfully",
            "articles_fetched": len(articles),
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error refreshing cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to refresh cache")

@news_router.get("/health")
async def news_health_check():
    """Check news service health"""
    try:
        # Try to fetch a small amount of news
        articles = await news_service.fetch_all_news(limit_per_source=1)
        
        return {
            "status": "healthy",
            "sources_available": len(news_service.sources),
            "articles_in_cache": sum(len(data["articles"]) for data in news_service.cache.values()),
            "cache_size": len(news_service.cache)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
    
@news_router.get("/article/{article_id}")
async def get_article_detail(
    article_id: str,
    fetch_content: bool = Query(False, description="Fetch full content")
):
    """Get detailed information about a specific article"""
    try:
        # Search for article in all cached news
        target_article = None
        for cache_key, cache_data in news_service.cache.items():
            if datetime.now().timestamp() - cache_data["timestamp"] < news_service.cache_timeout:
                for article in cache_data["articles"]:
                    if article.get("id") == article_id:
                        target_article = article
                        break
                if target_article:
                    break
        
        if not target_article:
            # Try to fetch fresh articles
            await news_service.fetch_all_news(limit_per_source=2)
            # Search again
            for cache_key, cache_data in news_service.cache.items():
                for article in cache_data["articles"]:
                    if article.get("id") == article_id:
                        target_article = article
                        break
                if target_article:
                    break
        
        if not target_article:
            raise HTTPException(status_code=404, detail="Article not found")
        
        # If fetch_content is True and we don't have content, try to fetch it
        if fetch_content and not target_article.get("content"):
            try:
                # You could implement article content scraping here
                pass
            except Exception as e:
                logger.debug(f"Could not fetch full content: {e}")
        
        # Prepare response with additional fields frontend expects
        response = {
            "id": target_article.get("id"),
            "title": target_article.get("title", ""),
            "description": target_article.get("description", ""),
            "summary": target_article.get("summary", ""),
            "content": target_article.get("content", ""),
            "link": target_article.get("link", ""),
            "published": target_article.get("published", ""),
            "source": target_article.get("source", ""),
            "source_id": target_article.get("source_id", ""),
            "category": target_article.get("category", ""),
            "image": target_article.get("image", ""),
            "top_image": target_article.get("image", ""),  # For modal
            "read_time": target_article.get("read_time", 1),
            "authors": [],  # Could extract from RSS if available
            "keywords": []  # Could generate from content
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error fetching article details: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch article details")