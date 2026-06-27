"""Phase 3: Telegram post generator — rule-based category + LLM content.

Converts cleaned AI news items into Telegram-ready structured posts.
- Category: deterministic keyword matching (no LLM)
- Post content: LLM per item (strict no-hallucination prompt)
- Confidence < 0.70 → skip
"""

import json
import re
from typing import Optional

import httpx

from .models import Article, ClassificationResult


# ── Constants ──────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.70

# Category keyword rules (order matters — first match wins)
CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("research", [
        "paper", "benchmark", "model training", "dataset", "experiment",
        "arxiv", "preprint", "research", "study", "findings", "methodology",
        "ablation", "state-of-the-art", "sota", "evaluation",
    ]),
    ("policy", [
        "regulation", "regulate", "law", "ban", "government", "eu ",
        "european union", "policy", "legislation", "compliance", "legal",
        "congress", "senate", "executive order", "sanction",
    ]),
    ("industry", [
        "funding", "acquisition", "acquire", "valuation", "ipo", "revenue",
        "layoff", "layoffs", "startup", "raises", "series a", "series b",
        "series c", "venture capital", "investment", "merger", "buyout",
    ]),
]

# LLM system prompt — strict no-hallucination
SYSTEM_PROMPT = """You are a technical content writer for a Telegram channel about AI news.

Your job is to convert a single news article into a concise Telegram post.

STRICT RULES:
- You MUST NOT add any information not explicitly present in the input content.
- You MUST NOT infer, speculate, or extrapolate beyond what is written.
- For "Why it matters": ONLY write a short paragraph if significance is EXPLICITLY stated in the content. If not, write EXACTLY: N/A
- No hype words (revolutionary, groundbreaking, game-changing, etc.)
- Neutral, technical tone
- High signal, low noise

OUTPUT FORMAT (use exactly this structure):

*<Clean Title>*

• <Key point 1>
• <Key point 2>
• <Key point 3>

🔍 *Why it matters:*
<1 short paragraph OR "N/A">

#<category will be provided>
🔗 <url will be provided>

RULES:
- Exactly 3 bullet points (or fewer only if content has fewer than 3 distinct facts)
- Each bullet: 1 sentence maximum
- Under 250 words total
- Title: no emojis, no clickbait, must reflect content accurately
- Use * for Telegram bold (not **)
- Use • for bullets (not - or *)
- No ---, no horizontal lines, no Markdown headings
- Do NOT include the category or URL — they will be added automatically"""


# ── Category Assignment (rule-based) ──────────────────────────────────────

def assign_category(content: str) -> str:
    """Assign category using deterministic keyword matching.

    Decision tree:
    - research: paper / benchmark / model training / dataset / experiment
    - policy: regulation / law / ban / government / EU / policy
    - industry: funding / acquisition / valuation / IPO / revenue / layoffs
    - product: default (none of the above)
    """
    content_lower = content.lower()

    for category, keywords in CATEGORY_RULES:
        for keyword in keywords:
            if keyword in content_lower:
                return category

    return "product"


# ── LLM Post Generation ──────────────────────────────────────────────────

async def generate_post(
    title: str,
    content: str,
    url: str,
    category: str,
    api_base: str,
    api_key: str,
    model: str,
) -> str:
    """Generate a Telegram post for a single news item using LLM."""

    user_prompt = f"""Convert this AI news article into a Telegram post.

Title: {title}
Content: {content}

Remember:
- Exactly 3 bullet points (1 sentence each)
- "Why it matters" — ONLY if explicitly stated in content, otherwise N/A
- No hallucination, no speculation
- Neutral technical tone

Generate the post now:"""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 500,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    async with httpx.AsyncClient(timeout=60, proxy="http://127.0.0.1:10808/") as client:
        resp = await client.post(
            f"{api_base}/chat/completions",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()

    data = resp.json()
    raw = data["choices"][0]["message"]["content"].strip()

    return raw


# ── Post Assembly ─────────────────────────────────────────────────────────

def _assemble_post(llm_output: str, category: str, url: str) -> str:
    """Assemble final post: LLM content + category tag + URL.

    Removes any category/URL the LLM might have added (shouldn't, but safety).
    """
    # Strip any category tag or URL the LLM might have hallucinated
    lines = llm_output.strip().split("\n")
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that look like category tags or URLs
        if stripped.startswith("#") and not stripped.startswith("🔍"):
            continue
        if stripped.startswith("🔗"):
            continue
        if stripped.startswith("http"):
            continue
        clean_lines.append(line)

    # Build final post
    post_body = "\n".join(clean_lines).strip()
    final_post = f"{post_body}\n\n#{category}\n🔗 {url}"

    return final_post


# ── Main Pipeline ─────────────────────────────────────────────────────────

async def generate_telegram_posts(
    items: list[dict],
    api_base: str,
    api_key: str,
    model: str,
) -> list[str]:
    """Generate Telegram posts for all items.

    Steps:
    1. Filter by confidence >= 0.70
    2. Assign category (rule-based)
    3. Generate post content (LLM per item)
    4. Assemble final post

    Returns list of formatted Telegram post strings.
    """
    posts = []

    for item in items:
        # Skip low confidence
        confidence = item.get("confidence", 0)
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        title = item.get("title", "")
        content = item.get("content", "")
        url = item.get("url", "")

        # Step 1: Category (rule-based)
        category = assign_category(content)

        # Step 2: Generate post (LLM)
        try:
            llm_output = await generate_post(
                title=title,
                content=content,
                url=url,
                category=category,
                api_base=api_base,
                api_key=api_key,
                model=model,
            )
        except Exception as e:
            # Skip items that fail
            print(f"  ✗ LLM error for '{title[:40]}': {e}")
            continue

        # Step 3: Assemble final post
        final_post = _assemble_post(llm_output, category, url)
        posts.append(final_post)

    return posts


def save_posts(posts: list[str], output_path: str) -> None:
    """Save posts to a text file with separators."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, post in enumerate(posts):
            f.write(post)
            if i < len(posts) - 1:
                f.write("\n\n" + "=" * 40 + "\n\n")
