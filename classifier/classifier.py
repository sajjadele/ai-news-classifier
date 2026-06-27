"""LLM-based article classifier."""

import json
import httpx
from .models import Article, ClassificationResult


CLASSIFICATION_PROMPT = """You are an AI news relevance classifier inside a news aggregation pipeline.

Your job is STRICT binary classification:

- Class 1: AI-RELEVANT NEWS
- Class 2: NOT AI-RELEVANT NEWS

---

# CRITICAL DEFINITION (VERY IMPORTANT)

An article is AI-RELEVANT ONLY IF:

✔️ It describes AI systems, models, or algorithms as the PRIMARY subject
✔️ It discusses AI companies (OpenAI, Anthropic, Google DeepMind, etc.) in a meaningful technical or business context
✔️ It reports on AI products, model releases, regulation, or deployment

---

# DO NOT MARK AS AI-RELEVANT IF:

❌ AI is only mentioned as a tool in passing
❌ AI is used incidentally (e.g. "used AI to write article", "AI helped analysis")
❌ It is a general business/personnel/news story involving an AI company but NOT about AI itself
❌ It is metaphorical or vague AI reference

---

# DECISION RULES (STRICT PRIORITY ORDER)

## 1. Primary AI subject test (HIGHEST PRIORITY)
If AI is not the main topic → NOT RELEVANT

## 2. Technical/business depth test
Must include at least one:
- model/system name
- AI product release
- AI infrastructure/deployment
- AI policy/regulation

## 3. Context dominance test
If removing AI mentions does NOT change meaning of article → NOT RELEVANT

---

# OUTPUT FORMAT (STRICT)

Return ONLY this JSON object, no other text:

{{
  "relevant": true | false,
  "confidence": 0.0 - 1.0,
  "reason": "short technical explanation grounded in text"
}}

---

# CONFIDENCE RULES

- 0.90–1.00 → explicit AI-centric article
- 0.70–0.89 → strong AI context but minor ambiguity
- 0.50–0.69 → borderline (prefer false unless strong evidence)

NEVER use high confidence unless AI is central topic.

---

# HARD CONSTRAINTS

- Do NOT assume AI relevance
- Do NOT infer missing context
- Do NOT classify based only on company name
- Do NOT be overly inclusive
- Prefer FALSE when uncertain

---

# OBJECTIVE

Maximize precision over recall.
It is better to miss weak AI articles than to incorrectly include non-AI articles.

---

## Input Article:
Title: {title}
Content: {content}

---

Return ONLY the JSON object. No other text.
"""


async def classify_article(
    article: Article,
    api_base: str,
    api_key: str,
    model: str = "gpt-3.5-turbo",
) -> ClassificationResult:
    """Classify a single article using an LLM."""

    prompt = CLASSIFICATION_PROMPT.format(
        title=article.title,
        content=article.content[:2000],  # Safety cap
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a precise classification module. Output ONLY valid JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 200,
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

    # Parse JSON from response (handle markdown code blocks)
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback if model doesn't return valid JSON
        result = {"relevant": False, "confidence": 0.0, "reason": f"Parse error: {raw[:100]}"}

    return ClassificationResult(
        relevant=result.get("relevant", False),
        confidence=float(result.get("confidence", 0.0)),
        reason=result.get("reason", "No reason provided"),
        article_title=article.title,
        article_url=article.url,
    )
