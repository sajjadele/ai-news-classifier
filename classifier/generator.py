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
    ("business", [
        "partnership", "deal", "contract", "revenue", "profit", "loss",
        "quarterly", "earnings", "market share", "growth", "strategy",
        "ceo", "cto", "executive", "hire", "hiring", "restructuring",
    ]),
]

# LLM system prompt — strict no-hallucination, Telegram-native formatting
SYSTEM_PROMPT = """You are a STRICT grounded content formatter and UX enhancer for a Telegram AI News channel.

You are NOT allowed to invent information.
You are NOT allowed to add facts.
You are NOT allowed to change meaning.

Your ONLY job is to improve readability and engagement of already-verified structured news items.

---

# OUTPUT FORMAT (STRICT)

For each item, output EXACTLY this format (use Telegram-native formatting):

🧠 *<Improved Title (still factual, no exaggeration)>*

• <Bullet 1: key factual point>
• <Bullet 2: key factual point>
• <Bullet 3: key factual point (optional if exists in input)>

🔍 *Why it matters:*
<ONLY derived implication from given content. Must be explicitly grounded. If not inferable, write: "Impact is limited to reported scope of the article.">

#<category will be provided>
🔗 <url will be provided>

---

# HARD CONSTRAINTS (NON-NEGOTIABLE)

## 1. NO NEW FACTS
- Do NOT introduce new entities
- Do NOT add technical details not present in input
- Do NOT assume motivations or hidden context

## 2. NO HALLUCINATED REASONING
- "Why it matters" must be directly inferable from input
- If not inferable → write: "Impact is limited to reported scope of the article."

## 3. TITLE RULES
- Must stay faithful to original meaning
- You MAY improve clarity, simplify, reorder words, remove redundancy
- You MUST NOT clickbait, emotional frame, or exaggerate
- Use Telegram bold: *Title* (with asterisks)

## 4. BULLET RULES
- Max 3 bullets
- Each bullet = one factual idea
- No repetition
- Use • for bullets

## 5. UX IMPROVEMENT ALLOWED (SAFE OPERATIONS)
You MAY: shorten sentences, improve readability, reorder for clarity, remove noise words, merge redundant phrases
You MAY NOT: add new claims, infer beyond text, expand scope

## 6. FORMATTING RULES (TELEGRAM-NATIVE)
- Use *text* for bold (NOT backticks, NOT **text**)
- Use • for bullets (NOT - or *)
- Do NOT use ---, horizontal lines, or Markdown headings
- Do NOT include the category or URL — they will be added automatically

---

# OBJECTIVE

Transform structured news into:
- readable
- clean
- Telegram-friendly
- fully grounded
- zero hallucination output

Generate the post now:"""


# ── Category Assignment (rule-based) ──────────────────────────────────────

def assign_category(content: str) -> str:
    """Assign category using deterministic keyword matching.

    Decision tree:
    - research: paper / benchmark / model training / dataset / experiment
    - policy: regulation / law / ban / government / EU / policy
    - industry: funding / acquisition / valuation / IPO / revenue / layoffs
    - business: partnership / deal / contract / revenue / profit / CEO
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
- Max 3 bullets (1 sentence each)
- "Why it matters" — ONLY if directly inferable from content, otherwise use fallback
- No hallucination, no speculation
- Telegram formatting: *bold*, • bullets
- Do NOT include category or URL — they will be added

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

    async with httpx.AsyncClient(timeout=60, proxy="http://127.0.0.1:10808/", verify=False) as client:
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

    Removes any category/URL the LLM might have added (safety).
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
        if stripped.startswith("Source:"):
            continue
        clean_lines.append(line)

    # Build final post — hierarchy: title → bullets → insight → category → source
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
