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

# LLM system prompt — Persian, production-grade, no hallucination
SYSTEM_PROMPT = """You are a technical news editor and content strategist for a Persian Telegram AI News channel.

Your job is to convert structured AI news data into high-quality Persian Telegram posts.

You do NOT simplify knowledge. You preserve technical depth while improving readability.

---

# OUTPUT FORMAT (STRICT)

For each item, output EXACTLY this format:

*<Persian translated title — bold, natural, human-readable>*
(<English original compressed title>)

• <Bullet 1: Fact — what happened>
• <Bullet 2: Mechanism — how it happened>
• <Bullet 3: Implication — why it matters>

🔍 چرا مهم است:
<One of: market implication / technical shift / strategic consequence — MUST be directly grounded in content>

#<category will be provided>
📖 مطالعه کامل مقاله در <source name will be provided>
<url will be provided>

---

# LANGUAGE RULES

- Title: Persian translation as main title (bold with *), original English title below in parentheses
- Bullet points: Persian, keep ALL technical terms in English (model names, company names, technical concepts like LLM, inference, fine-tuning, benchmark, etc.)
- Why it matters: Persian
- Category tags and URL: unchanged

---

# TITLE RULES

Format:
*<Persian translated title — bold>*
(<English original compressed title>)

- Persian part = natural, human-readable translation (bold with *)
- English part = original English title, compressed (in parentheses on next line)
- No clickbait, no exaggeration

---

# BULLET RULES

• 2–4 bullets max
• Each bullet = information-dense, ONE purpose:
  - Fact (what happened)
  - Mechanism (how it happened)
  - Implication (why it matters)
❌ Forbidden: generic statements, repetition of title, vague wording

---

# WHY IT MATTERS RULES

Must NOT be generic.
Must be one of:
- Market implication (e.g. "بازار عظیم آسیا برای آمریکاییها از دست میرود")
- Technical shift (e.g. "جابجایی از AI مبتنی بر مدل به AI مبتنی بر زیرساخت")
- Strategic consequence (e.g. "فشار تنظیمکنندهها استراتژی استقرار را تغییر میدهد")

MUST include the actual explanation, NOT just the label.
Example: ✅ "Strategic consequence — بازار آسیا ممکن است..."
Example: ❌ "Strategic consequence" (فقط لیبل بدون توضیح)

MUST be directly inferable from input content.
If not inferable → write based on what IS in the content (minimal, factual).
❌ Forbidden: "Impact is limited...", empty or neutral statements

---

# STYLE CONSTRAINTS

- No exaggeration
- No clickbait tone
- No emotional language
- No emojis except: 🧠 (title), 🔍 (why it matters), 📖 (source)

---

# HARD CONSTRAINTS (NON-NEGOTIABLE)

## NO NEW FACTS
- Do NOT introduce new entities
- Do NOT add technical details not present in input
- Do NOT assume motivations or hidden context

## NO HALLUCINATED REASONING
- "Why it matters" must be directly inferable from input
- Do NOT speculate beyond what is written

## QUALITY FILTER (SELF-CHECK)
Reject or rewrite if:
- too shallow
- no implication
- repetitive bullets
- missing technical anchor
- no clear "why it matters"

---

# OBJECTIVE

Do NOT summarize news.
Instead:
👉 Extract meaning
👉 Preserve technical depth
👉 Improve readability
👉 Add analytical value (ONLY from input content)

Generate the post now:"""


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
    category: str,
    source_name: str,
    api_base: str,
    api_key: str,
    model: str,
    proxy: str | None = None,
) -> str:
    """Generate a Persian Telegram post for a single news item using LLM."""

    user_prompt = f"""Convert this AI news article into a Persian Telegram post.

Title: {title}
Content: {content}
Category: {category}
Source: {source_name}

Rules:
- Primary language: Persian
- English only for technical terms
- Title: Persian hook + English reference in parentheses
- 2-4 bullets (Fact / Mechanism / Implication)
- "Why it matters" must be grounded in content
- Do NOT include category or URL — they will be added
- Telegram formatting: *bold* for title, • for bullets

Generate the post now:"""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
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

def _assemble_post(llm_output: str, category: str, source_name: str, url: str, time_ago: str = "") -> str:
    """Assemble final post: LLM content + category + source.

    Visual hierarchy: title → bullets → insight → category → source
    """
    # Strip any category/URL the LLM might have hallucinated
    lines = llm_output.strip().split("\n")
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") and not stripped.startswith("🔍"):
            continue
        if stripped.startswith("🔗"):
            continue
        if stripped.startswith("http"):
            continue
        if stripped.startswith("Source:"):
            continue
        if "مطالعه کامل" in stripped:
            continue
        clean_lines.append(line)

    # Build final post — hierarchy: title → bullets → insight → category → source
    post_body = "\n".join(clean_lines).strip()
    
    # Optional time line
    time_line = f"\n🕐 {time_ago}" if time_ago else ""
    
    final_post = f"{post_body}{time_line}\n\n#{category}\n📖 مطالعه کامل مقاله در {source_name}\n{url}"

    return final_post


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
        confidence = item.get("confidence", 0)
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        title = item.get("title", "")
        content = item.get("content", "")
        url = item.get("url", "")
        published_at = item.get("published_at")

        # Category (rule-based)
        category = assign_category(content)

        # Source name
        source_name = extract_source_name(url)

        # Generate post (LLM)
        try:
            llm_output = await generate_post(
                title=title,
                content=content,
                url=url,
                category=category,
                source_name=source_name,
                api_base=api_base,
                api_key=api_key,
                model=model,
                proxy=proxy,
            )
        except Exception as e:
            print(f"  ✗ LLM error for '{title[:40]}': {e}")
            continue

        # Assemble final post
        time_ago = _time_ago(published_at)
        final_post = _assemble_post(llm_output, category, source_name, url, time_ago)
        posts.append(final_post)

    return posts


def save_posts(posts: list[str], output_path: str) -> None:
    """Save posts to a text file with separators."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, post in enumerate(posts):
            f.write(post)
            if i < len(posts) - 1:
                f.write("\n\n" + "=" * 40 + "\n\n")
