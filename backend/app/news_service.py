# backend/app/news_service.py
"""
Legal News Aggregation Service (RSS-based)
Fetches real-time legal updates from RSS feeds
"""

import feedparser
import asyncio
import aiohttp
import re
import time
import logging
from datetime import datetime
from typing import List, Dict

logger = logging.getLogger(__name__)


class LegalNewsService:
    def __init__(self):
        # RSS-based legal news sources
        self.sources = {
            "scc_blog": {
                "name": "SCC Blog",
                "rss_url": "https://www.scconline.com/blog/feed/",
                "type": "rss"
            },
            "lawctopus": {
                "name": "Lawctopus",
                "rss_url": "https://www.lawctopus.com/feed/",
                "type": "rss"
            },
            "legal_bites": {
                "name": "Legal Bites",
                "rss_url": "https://www.legalbites.in/feed/",
                "type": "rss"
            },
            "indian_express": {
                "name": "The Indian Express - Legal",
                "rss_url": "https://indianexpress.com/section/law-and-policy/feed/",
                "type": "rss"
            },
            "live_law": {
                "name": "Live Law",
                "rss_url": "https://www.livelaw.in/rss/latest",
                "type": "rss"
            }
        }

        # Category labels
        self.categories = {
            "supreme_court": "Supreme Court Judgments",
            "high_court": "High Court Updates",
            "law_amendments": "Law Amendments",
            "analysis": "Legal Analysis",
            "career": "Legal Career"
        }

        # Cache (1 hour)
        self.cache = {}
        self.cache_timeout = 3600

        # Headers for RSS fetching
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

    # --------------------------------------------------
    # FETCH ALL NEWS (ASYNC)
    # --------------------------------------------------
    async def fetch_all_news(self, limit_per_source: int = 5) -> List[Dict]:
        try:
            tasks = [
                self.fetch_rss_news(source_id, limit_per_source)
                for source_id in self.sources
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            articles = []
            for result in results:
                if isinstance(result, list):
                    articles.extend(result)

            articles.sort(
                key=lambda x: x.get("published_timestamp", 0),
                reverse=True
            )

            return articles[:50]

        except Exception as e:
            logger.error(f"Error fetching all news: {e}")
            return []

    # --------------------------------------------------
    # FETCH FROM A SINGLE RSS SOURCE
    # --------------------------------------------------
    async def fetch_rss_news(self, source_id: str, limit: int) -> List[Dict]:
        source = self.sources.get(source_id)
        if not source:
            return []

        cache_key = f"rss_{source_id}_{limit}"
        cached = self.cache.get(cache_key)

        if cached and time.time() - cached["timestamp"] < self.cache_timeout:
            return cached["articles"]

        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(
                    source["rss_url"],
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:

                    if response.status != 200:
                        return []

                    feed = feedparser.parse(await response.text())

        except Exception as e:
            logger.warning(f"RSS fetch failed for {source_id}: {e}")
            return []

        articles = []

        for entry in feed.entries[:limit]:
            try:
                title = entry.get("title", "").strip()
                link = entry.get("link", "")
                description = entry.get("description", "")
                summary = entry.get("summary", "")

                published_timestamp = 0
                published = ""

                if entry.get("published_parsed"):
                    published_timestamp = time.mktime(entry.published_parsed)
                    published = datetime.fromtimestamp(
                        published_timestamp
                    ).isoformat()

                content = ""
                if entry.get("content") and isinstance(entry.content, list):
                    content = entry.content[0].get("value", "")
                elif description:
                    content = description

                article = {
                    "id": self.generate_article_id(link, title),
                    "title": title,
                    "description": description[:300],
                    "summary": summary[:200],
                    "content": content[:1000],
                    "link": link,
                    "published": published,
                    "published_timestamp": published_timestamp,
                    "source": source["name"],
                    "source_id": source_id,
                    "category": self.detect_category(
                        f"{title} {description} {content}"
                    ),
                    "image": self.extract_image_from_rss(entry),
                    "read_time": self.calculate_read_time(content)
                }

                if title and link:
                    articles.append(article)

            except Exception as e:
                logger.error(f"Error processing RSS entry: {e}")

        self.cache[cache_key] = {
            "articles": articles,
            "timestamp": time.time()
        }

        return articles

    # --------------------------------------------------
    # HELPERS
    # --------------------------------------------------
    def generate_article_id(self, link: str, title: str) -> str:
        import hashlib
        return hashlib.md5(f"{link}_{title}".encode()).hexdigest()[:16]

    def extract_image_from_rss(self, entry) -> str:
        if hasattr(entry, "media_content"):
            for media in entry.media_content:
                if media.get("type", "").startswith("image"):
                    return media.get("url", "")

        if hasattr(entry, "media_thumbnail") and entry.media_thumbnail:
            return entry.media_thumbnail[0].get("url", "")

        text_sources = [
            entry.get("summary", ""),
            entry.get("description", "")
        ]

        for text in text_sources:
            match = re.search(r'<img[^>]+src="([^"]+)"', text)
            if match:
                return match.group(1)

        return ""

    def detect_category(self, text: str) -> str:
        text = text.lower()

        patterns = {
            "Supreme Court Judgments": [r"supreme\s+court", r"\bsc\b"],
            "High Court Updates": [r"high\s+court", r"\bhc\b"],
            "Law Amendments": [r"amendment", r"bill", r"act"],
            "Legal Career": [r"internship", r"career", r"job"],
            "Legal Analysis": [r"analysis", r"opinion", r"editorial"]
        }

        for category, rules in patterns.items():
            if any(re.search(rule, text) for rule in rules):
                return category

        return "Legal News"

    def calculate_read_time(self, content: str) -> int:
        if not content:
            return 1
        words = len(re.sub(r"<[^>]+>", " ", content).split())
        return max(1, min(words // 200, 10))


# Global instance
news_service = LegalNewsService()
