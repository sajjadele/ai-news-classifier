"""News fetcher — RSS feeds and direct URLs."""

import httpx
import feedparser
from datetime import datetime
from .models import Article


async def fetch_rss(feed_url: str, limit: int = 10) -> list[Article]:
    """Fetch articles from an RSS/Atom feed."""
    async with httpx.AsyncClient(timeout=30, follow_redirects=True, proxy="http://127.0.0.1:10808/") as client:
        resp = await client.get(feed_url)
        resp.raise_for_status()

    feed = feedparser.parse(resp.text)
    articles = []

    for entry in feed.entries[:limit]:
        # Extract content — prefer full content, fall back to summary
        content = ""
        if hasattr(entry, "content") and entry.content:
            content = entry.content[0].get("value", "")
        elif hasattr(entry, "summary"):
            content = entry.summary or ""

        # Strip HTML tags (basic)
        import re
        content = re.sub(r"<[^>]+>", " ", content)
        content = re.sub(r"\s+", " ", content).strip()

        if not content:
            continue

        pub_date = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            try:
                pub_date = datetime(*entry.published_parsed[:6])
            except Exception:
                pass

        articles.append(Article(
            title=entry.get("title", "Untitled"),
            content=content[:3000],  # Cap content length
            url=entry.get("link"),
            source=feed_url,
            published_at=pub_date,
        ))

    return articles


async def fetch_url(url: str) -> Article:
    """Fetch a single article from a URL."""
    async with httpx.AsyncClient(timeout=30, follow_redirects=True, proxy="http://127.0.0.1:10808/") as client:
        resp = await client.get(url)
        resp.raise_for_status()

    text = resp.text
    # Basic HTML stripping
    import re
    # Extract title
    title_match = re.search(r"<title>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
    title = title_match.group(1).strip() if title_match else url

    # Strip tags
    content = re.sub(r"<[^>]+>", " ", text)
    content = re.sub(r"\s+", " ", content).strip()

    return Article(
        title=title,
        content=content[:3000],
        url=url,
        source="direct",
    )


async def fetch_multiple_urls(urls: list[str]) -> list[Article]:
    """Fetch multiple URLs concurrently."""
    import asyncio
    tasks = [fetch_url(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    articles = []
    for r in results:
        if isinstance(r, Article):
            articles.append(r)
        # Skip errors silently
    return articles
