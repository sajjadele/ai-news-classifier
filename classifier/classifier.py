"""LLM-based article classifier."""

import json
import httpx
from .models import Article, ClassificationResult


CLASSIFICATION_PROMPT = """You are a high-precision classification module in an AI news processing system.

Your task is to determine whether the given article is relevant to Artificial Intelligence (AI) or not.

## Definition of AI relevance:
Include ONLY content related to:
- Machine Learning (ML)
- Deep Learning
- Large Language Models (LLMs)
- Generative AI
- AI research papers or breakthroughs
- AI products, companies, or model releases
- AI infrastructure (training, inference, GPUs) IF directly related
- AI regulation or policy IF directly impacts AI systems

Exclude:
- General tech news without AI component
- Marketing content without technical substance
- Hardware news unrelated to AI workloads
- Crypto, blockchain, general software updates (unless AI-specific)

---

## Input Article:
Title: {title}
Content: {content}

---

## Output Rules (STRICT JSON ONLY):

Return ONLY this JSON object, no other text:

{{
  "relevant": true/false,
  "confidence": 0.0 to 1.0,
  "reason": "short technical justification"
}}

---

## Decision Rules:
- If uncertain → choose false unless AI relevance is explicit
- Prioritize precision over recall
- Do NOT assume AI relevance from vague keywords like "smart", "automation", "future"
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
