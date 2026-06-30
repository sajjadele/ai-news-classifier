"""Batch test: classify 30 articles from multiple feeds and report results."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from classifier.fetcher import fetch_rss
from classifier.classifier import classify_article
from cli import load_config


async def main():
    config = load_config()
    api_base = config["api_base"]
    api_key = config["api_key"]
    model = config["model"]
    proxy = config.get("proxy")

    feeds = [
        ("TechCrunch AI", "https://techcrunch.com/category/artificial-intelligence/feed/"),
        ("The Verge AI", "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml"),
        ("MIT Tech Review", "https://www.technologyreview.com/topic/artificial-intelligence/feed"),
    ]

    all_articles = []

    # Fetch 10 from each feed
    for name, url in feeds:
        print(f"📥 Fetching from {name}...")
        try:
            articles = await fetch_rss(url, limit=10, proxy=proxy)
            for a in articles:
                a.source = name  # tag with friendly name
            all_articles.extend(articles)
            print(f"   ✓ {len(articles)} articles")
        except Exception as e:
            print(f"   ✗ Error: {e}")

    print(f"\n📊 Total fetched: {len(all_articles)}")
    print(f"{'='*80}")

    # Classify all
    results = []
    for i, article in enumerate(all_articles):
        print(f"  [{i+1}/{len(all_articles)}] {article.title[:60]}...", flush=True)
        result = await classify_article(article, i, api_base, api_key, model, proxy=proxy)
        results.append((article, result))

    # ── Report ──────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"{'#':<3} {'Relevant':^10} {'Conf':^6} {'Source':<16} {'Title':<50} {'Reason'}")
    print(f"{'='*80}")

    relevant_count = 0
    not_relevant_count = 0
    errors = 0

    for i, (article, result) in enumerate(results, 1):
        if result.error:
            tag = "⚠ ERR"
            errors += 1
        elif result.relevant:
            tag = "✅ YES"
            relevant_count += 1
        else:
            tag = "❌ NO"
            not_relevant_count += 1

        conf = f"{result.confidence:.2f}"
        source = (article.source or "?")[:15]
        title = article.title[:48]
        reason = result.reason[:50] if result.reason else ""

        print(f"{i:<3} {tag:^10} {conf:^6} {source:<16} {title:<50} {reason}")

    print(f"\n{'='*80}")
    print(f"SUMMARY: {len(results)} articles | ✅ {relevant_count} relevant | ❌ {not_relevant_count} not relevant | ⚠ {errors} errors")
    print(f"Avg confidence: {sum(r.confidence for _, r in results)/len(results):.2f}")


if __name__ == "__main__":
    asyncio.run(main())
