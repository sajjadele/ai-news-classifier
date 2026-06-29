"""Phase 2: Post-classification processor — dedup, noise removal, normalization.

Completely rule-based and deterministic. No LLM calls.
"""

import re
import unicodedata
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from urllib.parse import urlparse

from .models import Article, ClassificationResult


# ── Constants ──────────────────────────────────────────────────────────────

DEDUP_TITLE_THRESHOLD = 0.82
DEDUP_TIME_WINDOW_HOURS = 48
MIN_CONTENT_LENGTH = 80

NOISE_KEYWORDS = [
    "sponsored",
    "subscribe",
    "newsletter",
    "sign up",
]

# Emoji pattern — matches most Unicode emoji ranges
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"  # dingbats
    "\U000024C2-\U0001F251"  # enclosed characters
    "\U0001f926-\U0001f937"  # supplemental
    "\U00010000-\U0010ffff"  # supplementary
    "\u2640-\u2642"          # gender
    "\u2600-\u2B55"          # misc symbols
    "\u200d"                 # zero width joiner
    "\u23cf"                 # eject symbol
    "\u23e9-\u23f3"          # media symbols
    "\u23f8-\u23fa"          # media symbols
    "\ufe0f"                 # variation selector
    "]+",
    flags=re.UNICODE,
)


# ── Helper: Wrapped result with metadata ──────────────────────────────────

class ProcessedItem:
    """Wraps article + classification for processing pipeline."""

    def __init__(self, article: Article, classification: ClassificationResult):
        self.article = article
        self.classification = classification

    def to_dict(self) -> dict:
        return {
            "title": self.article.title,
            "content": self.article.content,
            "url": self.article.url,
            "source": self.article.source,
            "published_at": self.article.published_at.isoformat() if self.article.published_at else None,
            "confidence": self.classification.confidence,
            "relevant": self.classification.relevant,
            "error": self.classification.error,
        }


# ── 1. Deduplication ──────────────────────────────────────────────────────

def _extract_domain(url: str | None) -> str:
    """Extract domain from URL."""
    if not url:
        return ""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _title_similarity(a: str, b: str) -> float:
    """SequenceMatcher ratio between two titles (case-insensitive)."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _within_time_window(item_a: ProcessedItem, item_b: ProcessedItem, hours: int = DEDUP_TIME_WINDOW_HOURS) -> bool:
    """Check if two items were published within N hours of each other."""
    a, b = item_a.article.published_at, item_b.article.published_at
    if not a or not b:
        return True  # If unknown, assume possible duplicate
    return abs((a - b).total_seconds()) < hours * 3600


def _pick_keeper(a: ProcessedItem, b: ProcessedItem) -> ProcessedItem:
    """Choose which item to keep: higher confidence → longer content."""
    if a.classification.confidence != b.classification.confidence:
        return a if a.classification.confidence > b.classification.confidence else b
    # Tie-break: longer content
    return a if len(a.article.content) >= len(b.article.content) else b


def deduplicate(items: list[ProcessedItem]) -> list[ProcessedItem]:
    """Remove duplicate items describing the same real-world event.

    Uses title similarity (SequenceMatcher >= 0.82) + time window (48h).
    Keeps the item with higher confidence, then longer content.
    """
    if len(items) <= 1:
        return items

    to_remove: set[int] = set()

    for i in range(len(items)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(items)):
            if j in to_remove:
                continue

            a, b = items[i], items[j]

            # Check title similarity
            sim = _title_similarity(a.article.title, b.article.title)
            if sim < DEDUP_TITLE_THRESHOLD:
                continue

            # Check time window
            if not _within_time_window(a, b):
                continue

            # Same event — pick keeper
            loser_idx = j if _pick_keeper(a, b) is a else i
            to_remove.add(loser_idx)

    return [item for idx, item in enumerate(items) if idx not in to_remove]


# ── 2. Noise Removal ──────────────────────────────────────────────────────

def _is_noise(item: ProcessedItem) -> bool:
    """Check if an item is noise (content too short, empty, or promotional).

    Conservative: returns True ONLY when clearly noise.
    """
    content = (item.article.content or "").strip()

    # Empty or null
    if not content:
        return True

    # Too short
    if len(content) < MIN_CONTENT_LENGTH:
        return True

    # Promotional keywords (case-insensitive)
    content_lower = content.lower()
    for keyword in NOISE_KEYWORDS:
        if keyword in content_lower:
            return True

    return False


def remove_noise(items: list[ProcessedItem]) -> list[ProcessedItem]:
    """Remove items that are clearly noise.

    Conservative approach: only removes when clearly shallow, empty, or promotional.
    """
    return [item for item in items if not _is_noise(item)]


# ── 3. Title Normalization ────────────────────────────────────────────────

def _count_emojis(text: str) -> int:
    """Count individual emoji characters in text."""
    return sum(len(m) for m in EMOJI_PATTERN.findall(text)) if text else 0


def _remove_emojis(text: str) -> str:
    """Remove all emoji characters from text."""
    return EMOJI_PATTERN.sub("", text).strip()


def _clean_title(title: str) -> str:
    """Clean title: remove excessive emojis, normalize whitespace.

    - If more than 1 emoji → remove all emojis
    - Normalize whitespace (multiple spaces → single)
    - Keep meaning unchanged
    """
    if not title:
        return title

    emoji_count = _count_emojis(title)
    if emoji_count > 1:
        title = _remove_emojis(title)

    # Normalize whitespace
    title = re.sub(r"\s+", " ", title).strip()

    return title


def normalize_titles(items: list[ProcessedItem]) -> list[ProcessedItem]:
    """Normalize titles: remove excessive emojis, clean whitespace.

    Content is NOT modified.
    """
    for item in items:
        item.article.title = _clean_title(item.article.title)
    return items


# ── Main Pipeline ─────────────────────────────────────────────────────────

def process(
    articles: list[Article],
    classifications: list[ClassificationResult],
) -> list[dict]:
    """Run the full Phase 2 pipeline.

    Steps (strict order):
    1. Deduplication
    2. Noise removal
    3. Title normalization

    Returns list of dicts in the standard output schema.
    """
    # Build wrapped items
    items = [
        ProcessedItem(article=art, classification=cls)
        for art, cls in zip(articles, classifications)
    ]

    # Step 1: Deduplicate
    items = deduplicate(items)

    # Step 2: Noise removal
    items = remove_noise(items)

    # Step 3: Normalize titles
    items = normalize_titles(items)

    # Output
    return [item.to_dict() for item in items]
