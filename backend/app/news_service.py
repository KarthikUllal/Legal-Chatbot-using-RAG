# backend/app/news_service.py
"""
Legal News Scraper Service
Fetches real-time legal updates from various sources
"""
import requests
from bs4 import BeautifulSoup
import feedparser
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging
import json
import asyncio
import aiohttp
import re
import time

logger = logging.getLogger(__name__)

class LegalNewsService:
    # Initialize news service with sources, categories, cache and headers
    def __init__(self):
        # Define legal news sources (RSS feeds of legal websites)
        self.sources = {
            "scc_blog": {
                "name": "SCC Blog",
                "rss_url": "https://www.scconline.com/blog/feed/",
                "website_url": "https://www.scconline.com/blog/",
                "type": "rss"
            },
            "lawctopus": {
                "name": "Lawctopus",
                "rss_url": "https://www.lawctopus.com/feed/",
                "website_url": "https://www.lawctopus.com/",
                "type": "rss"
            },
            "legal_bites": {
                "name": "Legal Bites",
                "rss_url": "https://www.legalbites.in/feed/",
                "website_url": "https://www.legalbites.in/",
                "type": "rss"
            },
            "indian_express": {
                "name": "The Indian Express - Legal",
                "rss_url": "https://indianexpress.com/section/law-and-policy/feed/",
                "website_url": "https://indianexpress.com/section/law-and-policy/",
                "type": "rss"
            },
            "live_law": {
                "name": "Live Law",
                "rss_url": "https://www.livelaw.in/rss/latest",
                "website_url": "https://www.livelaw.in/",
                "type": "rss"
            }
        }
        
        # Define legal news categories for classification
        self.categories = {
            "supreme_court": "Supreme Court Judgments",
            "high_court": "High Court Updates", 
            "law_amendments": "Law Amendments",
            "legal_news": "Legal News",
            "analysis": "Legal Analysis",
            "career": "Legal Career"
        }
        
        # Cache for news articles (1 hour expiry) to avoid repeated API calls
        self.cache = {}
        self.cache_timeout = 3600
        
        # Browser headers to mimic real browser requests (avoid blocking)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }
    

    # Fetch news from all sources simultaneously using async calls
    async def fetch_all_news(self, limit_per_source: int = 5) -> List[Dict]:
        """Fetch news from all sources asynchronously"""
        try:
            tasks = []
            
            # Create async tasks for each RSS source
            for source_id, source_info in self.sources.items():
                if source_info["type"] == "rss":
                    tasks.append(self.fetch_rss_news(source_id, limit_per_source))
            
            # Run all tasks concurrently (fetch multiple sources at once)
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results from all sources
            all_articles = []
            for result in results:
                if isinstance(result, list):
                    all_articles.extend(result)
                elif result:
                    logger.error(f"Unexpected result type: {type(result)}")
            
            # Sort by date (newest first) for better user experience
            all_articles.sort(key=lambda x: x.get("published_timestamp", 0), reverse=True)
            
            # Limit total articles to prevent overwhelming users
            return all_articles[:50]
            
        except Exception as e:
            logger.error(f"Error fetching all news: {e}")
            return []
    

    # Fetch news from a single RSS feed source
    async def fetch_rss_news(self, source_id: str, limit: int = 5) -> List[Dict]:
        """Fetch news from RSS feeds"""
        try:
            source = self.sources.get(source_id)
            if not source:
                return []
            
            # Check cache first - return cached data if still valid (1 hour)
            cache_key = f"rss_{source_id}_{limit}"
            if cache_key in self.cache:
                cache_data = self.cache[cache_key]
                if datetime.now().timestamp() - cache_data["timestamp"] < self.cache_timeout:
                    return cache_data["articles"]
            
            # Fetch RSS feed with timeout to prevent hanging
            try:
                async with aiohttp.ClientSession(headers=self.headers) as session:
                    async with session.get(source["rss_url"], timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status != 200:
                            logger.warning(f"Failed to fetch RSS from {source_id}: HTTP {response.status}")
                            return []
                        
                        # Parse RSS feed content
                        feed_content = await response.text()
                        feed = feedparser.parse(feed_content)
            except Exception as e:
                logger.warning(f"Failed to fetch RSS from {source_id}: {e}")
                return []
            
            articles = []
            
            # Process each RSS entry (news article)
            for entry in feed.entries[:limit]:
                try:
                    # Extract basic article info from RSS
                    title = entry.get("title", "").strip()
                    link = entry.get("link", "")
                    description = entry.get("description", "")
                    summary = entry.get("summary", "")
                    
                    # Parse publication date from RSS (convert to timestamp)
                    published = ""
                    published_timestamp = 0
                    
                    if entry.get("published"):
                        published = entry.get("published")
                        try:
                            published_timestamp = time.mktime(entry.get("published_parsed", time.localtime()))
                            published = datetime.fromtimestamp(published_timestamp).isoformat()
                        except:
                            pass
                    
                    # Get full content (prefer content field, fallback to description)
                    content = ""
                    if entry.get("content"):
                        if isinstance(entry.content, list) and len(entry.content) > 0:
                            content = entry.content[0].get("value", "")
                    elif description:
                        content = description
                    
                    # Extract featured image from article
                    image = self.extract_image_from_rss(entry)
                    
                    # Auto-detect legal category based on text content
                    category = self.detect_category(title + " " + description + " " + content)
                    
                    # Calculate estimated reading time (200 words/minute)
                    read_time = self.calculate_read_time(content)
                    
                    # Build complete article object with all metadata
                    article = {
                        "id": self.generate_article_id(link, title),
                        "title": title,
                        "description": description[:300] + "..." if len(description) > 300 else description,
                        "summary": summary[:200] + "..." if len(summary) > 200 else summary,
                        "content": content[:1000] + "..." if len(content) > 1000 else content,
                        "link": link,
                        "published": published,
                        "published_timestamp": published_timestamp,
                        "source": source["name"],
                        "source_id": source_id,
                        "category": category,
                        "image": image,
                        "read_time": read_time
                    }
                    
                    # Only add valid articles (must have title and link)
                    if article["title"] and article["link"]:
                        articles.append(article)
                    
                except Exception as e:
                    logger.error(f"Error processing RSS entry from {source_id}: {e}")
                    continue
            
            # Cache the results for 1 hour to avoid repeated fetching
            self.cache[cache_key] = {
                "articles": articles,
                "timestamp": datetime.now().timestamp()
            }
            
            logger.info(f"Fetched {len(articles)} articles from {source_id}")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching RSS from {source_id}: {e}")
            return []
    

    # Generate unique ID for each article using MD5 hash
    def generate_article_id(self, link: str, title: str) -> str:
        """Generate a unique ID for an article"""
        import hashlib
        unique_string = f"{link}_{title}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:16]
    
    # Extract image URL from RSS entry (multiple possible locations)
    def extract_image_from_rss(self, entry) -> str:
        """Extract image URL from RSS entry"""
        try:
            # Check for media:content (RSS 2.0 media extension)
            if hasattr(entry, 'media_content'):
                for media in entry.media_content:
                    if media.get('type', '').startswith('image'):
                        return media.get('url', '')
            
            # Check for media:thumbnail (common for news feeds)
            if hasattr(entry, 'media_thumbnail') and entry.media_thumbnail:
                return entry.media_thumbnail[0].get('url', '')
            
            # Check for enclosure (podcast-style image attachments)
            if hasattr(entry, 'enclosures') and entry.enclosures:
                for enc in entry.enclosures:
                    if enc.get('type', '').startswith('image'):
                        return enc.get('href', '')
            
            # Check links array for image links
            if hasattr(entry, 'links'):
                for link in entry.links:
                    if link.get('type', '').startswith('image'):
                        return link.get('href', '')
            
            # Finally, search for img tags in HTML content
            content = entry.get('content', [{}])[0].get('value', '') if entry.get('content') else ''
            summary = entry.get('summary', '')
            
            for text in [content, summary, entry.get('description', '')]:
                if text:
                    img_match = re.search(r'<img[^>]+src="([^"]+)"', text, re.IGNORECASE)
                    if img_match:
                        return img_match.group(1)
            
            return ""
        except Exception as e:
            logger.debug(f"Error extracting image: {e}")
            return ""
    

    # Auto-categorize articles based on keywords (legal-specific)
    def detect_category(self, text: str) -> str:
        """Detect category based on text content"""
        if not text:
            return "Legal News"
        
        text_lower = text.lower()
        
        # Legal-specific regex patterns for each category
        category_patterns = {
            "supreme_court": [
                r"supreme\s+court", r"\bsc\b", r"justice\s+\w+", r"bench\s+of",
                r"supreme\s+court\s+judgment", r"sc\s+judgment", r"apex\s+court"
            ],
            "high_court": [
                r"high\s+court", r"\bhc\b", r"delhi\s+high\s+court", r"bombay\s+high\s+court",
                r"madras\s+high\s+court", r"calcutta\s+high\s+court", r"kerala\s+high\s+court"
            ],
            "law_amendments": [
                r"amendment", r"bill", r"act\s+of", r"parliament", r"ordinance",
                r"notification", r"gazette", r"law\s+commission"
            ],
            "career": [
                r"career", r"internship", r"job", r"vacancy", r"recruitment",
                r"clerk", r"llm", r"clat", r"exam", r"admission"
            ],
            "analysis": [
                r"analysis", r"opinion", r"view", r"perspective", r"commentary",
                r"editorial", r"interpretation", r"case\s+comment"
            ]
        }
        
        # Check text against each pattern, return first match
        for category_id, patterns in category_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return self.categories.get(category_id, "Legal News")
        
        return "Legal News"
    

    # Calculate reading time based on word count (200 words/minute)
    def calculate_read_time(self, content: str) -> int:
        """Calculate estimated read time in minutes"""
        if not content:
            return 1
        
        # Remove HTML tags to count only text
        import re
        text_only = re.sub(r'<[^>]+>', ' ', content)
        
        # Count words in clean text
        words = len(text_only.split())
        
        # Calculate minutes (minimum 1, maximum 10)
        read_time = max(1, int(words / 200))
        return min(read_time, 10)
    

    # Filter articles by specific legal category (from cache)
    def get_legal_updates_by_category(self, category: str) -> List[Dict]:
        """Get legal updates filtered by category"""
        all_articles = []
        # Search through cache for articles matching the category
        for cache_key, cache_data in self.cache.items():
            if datetime.now().timestamp() - cache_data["timestamp"] < self.cache_timeout:
                for article in cache_data["articles"]:
                    if article.get("category", "") == category:
                        all_articles.append(article)
        
        # Return newest first, limited to 20 articles
        return sorted(all_articles, key=lambda x: x.get("published_timestamp", 0), reverse=True)[:20]
    

    # Search articles by keyword (title, description, content)
    def search_legal_news(self, query: str) -> List[Dict]:
        """Search legal news by keyword"""
        results = []
        query_lower = query.lower()
        
        # Search through cached articles
        for cache_key, cache_data in self.cache.items():
            if datetime.now().timestamp() - cache_data["timestamp"] < self.cache_timeout:
                for article in cache_data["articles"]:
                    # Search in title, description, and content
                    if (query_lower in article.get("title", "").lower() or 
                        query_lower in article.get("description", "").lower() or
                        query_lower in article.get("content", "").lower()):
                        results.append(article)
        
        # Return newest matching articles first
        return sorted(results, key=lambda x: x.get("published_timestamp", 0), reverse=True)

# Create global instance for easy access across the application
news_service = LegalNewsService()