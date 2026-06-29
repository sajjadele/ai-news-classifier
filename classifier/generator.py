"""Phase 3: Telegram post generator — Persian, production-grade.

Converts cleaned AI news items into high-quality Persian Telegram posts.
- Category: deterministic keyword matching (no LLM)
- Post content: LLM per item (strict no-hallucination, Persian output)
- Confidence < 0.70 → skip
"""

import json
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx


# ── Constants ──────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.70

# Source name mapping (domain → friendly Persian name)
SOURCE_NAMES = {
    "techcrunch.com": "TechCrunch",
    "theverge.com": "The Verge",
    "technologyreview.com": "MIT Technology Review",
    "arstechnica.com": "Ars Technica",
    "wired.com": "WIRED",
    "venturebeat.com": "VentureBeat",
    "thenextweb.com": "The Next Web",
    "engadget.com": "Engadget",
    "zdnet.com": "ZDNet",
    "theregister.com": "The Register",
    "bbc.com": "BBC",
    "reuters.com": "Reuters",
    "bloomberg.com": "Bloomberg",
    "nytimes.com": "New York Times",
    "theguardian.com": "The Guardian",
}

# New taxonomy keyword rules
CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("Research", [
        "paper", "benchmark", "model training", "dataset", "experiment",
        "arxiv", "preprint", "research", "study", "findings", "methodology",
        "ablation", "state-of-the-art", "sota", "evaluation", "peer-reviewed",
    ]),
    ("AIPolicy", [
        "regulation", "regulate", "law", "ban", "government", "eu ",
        "european union", "policy", "legislation", "compliance", "legal",
        "congress", "senate", "executive order", "sanction", "antitrust",
    ]),
    ("Funding", [
        "funding", "raises", "series a", "series b", "series c",
        "venture capital", "investment", "seed round", "valuation",
    ]),
    ("AIHardware", [
        "chip", "gpu", "tpu", "hardware", "semiconductor", "nvidia",
        "amd", "intel", "custom silicon", "asic", "accelerator",
        "data center", "server",
    ]),
    ("BigTech", [
        "google", "microsoft", "apple", "amazon", "meta", "facebook",
        "alphabet", "openai", "anthropic", "deepmind",
    ]),
    ("Startups", [
        "startup", "founder", "launch", "yc", "accelerator", "incubator",
    ]),
    ("Infrastructure", [
        "cloud", "api", "deployment", "inference", "training infrastructure",
        "mlops", "devops", "kubernetes", "docker", "scaling",
    ]),
]

SYSTEM_PROMPT = """You are a technical content writer for a Persian Telegram channel about AI news.

Your job is to convert a single news article into a Telegram post in Persian using HTML formatting.

STRICT RULES:
- Write in Persian. Keep ALL technical terms, model names, and company names in English.
- You MUST NOT add any information not explicitly present in the input content.
- Use Telegram HTML formatting ONLY: <b>bold</b>, <i>italic</i>, <blockquote>...</blockquote>, <a href="URL">text</a>
- Never use Markdown (* ** _ `). Never use - or * for bullets. Always use • character.

OUTPUT FORMAT (follow exactly, no deviation):

<b>[Persian title — accurate, no clickbait, no emojis]</b>

<blockquote>
- [Key point 1 in Persian, technical terms in English]
- [Key point 2 in Persian, technical terms in English]
- [Key point 3 in Persian, technical terms in English]
</blockquote>

💡 <i>تحلیل: [1 sentence — model's analysis of significance. Clearly the model's perspective.]</i>

#[category]  #[entity1]  #[entity2]

🕐 [X روز پیش | Source name]
📖 <a href="[EXACT_URL_FROM_INPUT]">مطالعه کامل خبر</a>

➖➖➖
📢 برای اطلاع از آخرین اخبار هوش مصنوعی کانال ما را دنبال کنید
@Ai_daily_news_fa

HASHTAG RULES:
- All hashtags on ONE line, separated by two spaces.
- Always include ONE category hashtag:
  #پژوهش (paper, benchmark, dataset, experiment)
  #سیاست (regulation, law, ban, government, EU)
  #صنعت (funding, acquisition, valuation, IPO, layoffs)
  #محصول (release, launch, API, update — default)
- Add entity hashtags for mentioned companies/topics:
  #OpenAI #Anthropic #Google #Meta #Microsoft #Apple #Nvidia
  #LLM #AGI #Agents #ImageAI #HardwareAI
- Maximum 4 hashtags total.

URL RULES:
- Use the EXACT url from the input. Do not modify or shorten it.
- Place it inside <a href="URL">مطالعه کامل خبر</a>

DAYS AGO RULES:
- Calculate from published_at if available.
- Format: X روز پیش (e.g. ۱ روز پیش, ۲ روز پیش)
- If published today: امروز
- If published_at is None: write: تازه منتشر شده

ANALYSIS RULES:
- تحلیل must be 1 sentence maximum.
- It is the model's reasoning, not a stated fact.
- Never fabricate facts. Base analysis only on article content."""


# ── Time Display ─────────────────────────────────────────────────────────

def _time_ago(published_at: str | None) -> str:
    """Convert ISO datetime string to human-readable Persian time ago.
    
    Examples: "چند دقیقه پیش", "۳ ساعت پیش", "۲ روز پیش"
    """
    if not published_at:
        return ""
    try:
        pub = datetime.fromisoformat(published_at)
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = now - pub
        
        if delta.days > 1:
            return f"{delta.days} روز پیش"
        elif delta.days == 1:
            return "دیروز"
        else:
            hours = delta.seconds // 3600
            if hours >= 1:
                return f"{hours} ساعت پیش"
            return "چند دقیقه پیش"
    except Exception:
        return ""


# ── Category Assignment (rule-based) ──────────────────────────────────────

def assign_category(content: str) -> str:
    """Assign category using deterministic keyword matching."""
    content_lower = content.lower()

    for category, keywords in CATEGORY_RULES:
        for keyword in keywords:
            if keyword in content_lower:
                return category

    return "AIModels"


# ── Source Name Extraction ────────────────────────────────────────────────

def extract_source_name(url: str | None) -> str:
    """Extract friendly source name from URL."""
    if not url:
        return "منبع اصلی"
    try:
        domain = urlparse(url).netloc.lower()
        # Remove www.
        domain = domain.replace("www.", "")
        # Check mapping
        for key, name in SOURCE_NAMES.items():
            if key in domain:
                return name
        # Fallback: capitalize domain without TLD
        parts = domain.split(".")
        if len(parts) >= 2:
            return parts[-2].capitalize()
        return domain
    except Exception:
        return "منبع اصلی"


# ── LLM Post Generation ──────────────────────────────────────────────────

async def generate_post(
    title: str,
    content: str,
    url: str,
    source: str,
    published_at: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    proxy: str | None = None,
) -> str:
    """Generate a Persian Telegram post for a single news item using LLM."""

    user_prompt = f"""Convert this AI news article into a Persian Telegram post.

Title: {title}
Content: {content}
URL: {url}
Source: {source}
Published: {published_at}

Follow the exact output format. Use the exact URL provided. Calculate days ago from published date.
Generate the post now:"""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.5,
        "max_tokens": 600,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds

    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=60, proxy=proxy, verify=False) as client:
                resp = await client.post(
                    f"{api_base}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
        except httpx.HTTPStatusError as e:
            if attempt < MAX_RETRIES - 1:
                import asyncio
                await asyncio.sleep(RETRY_DELAY)
                continue
            raise

    # Should never reach here, but just in case
    raise RuntimeError("All retries failed")


# ── Post Assembly ─────────────────────────────────────────────────────────

def _assemble_post(llm_output: str, category: str, url: str) -> str:
    """Assemble final post. Pass-through — LLM generates the complete post."""
    return llm_output.strip()


# ── Main Pipeline ─────────────────────────────────────────────────────────

async def generate_telegram_posts(
    items: list[dict],
    api_base: str,
    api_key: str,
    model: str,
    proxy: str | None = None,
) -> list[str]:
    """Generate Telegram posts for all items.

    Steps:
    1. Filter by confidence >= 0.70
    2. Assign category (rule-based)
    3. Extract source name
    4. Generate post content (LLM per item)
    5. Assemble final post

    Returns list of formatted Telegram post strings.
    """
    posts = []

    for item in items:
        if not item.get("relevant", False):
            continue
        if item.get("error"):
            continue
        confidence = item.get("confidence", 0)
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        title = item.get("title", "")
        content = item.get("content", "")
        url = item.get("url", "")
        published_at = item.get("published_at")

        # Source name
        source_name = extract_source_name(url)

        # Generate post (LLM) — the LLM handles category, hashtags, and timestamp
        try:
            llm_output = await generate_post(
                title=title,
                content=content,
                url=url,
                source=source_name,
                published_at=published_at,
                api_base=api_base,
                api_key=api_key,
                model=model,
                proxy=proxy,
            )
        except Exception as e:
            print(f"  ✗ LLM error for '{title[:40]}': {e}")
            continue

        # Assemble final post (pass-through)
        final_post = _assemble_post(llm_output, category="", url=url)
        posts.append(final_post)

    return posts


def save_posts(posts: list[str], output_path: str) -> None:
    """Save posts to a text file with separators."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, post in enumerate(posts):
            f.write(post)
            if i < len(posts) - 1:
                f.write("\n\n" + "=" * 40 + "\n\n")
