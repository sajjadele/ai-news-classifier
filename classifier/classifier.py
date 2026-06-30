"""LLM-based article classifier — strict relevance classification."""

import json
import httpx
from .models import Article, ClassificationResult


SYSTEM_PROMPT = """You are a strict AI News Relevance Classifier.

Your job is to decide whether a news article is PRIMARILY about Artificial Intelligence.

========================

PRIMARY RULE

Mark an article as relevant ONLY if Artificial Intelligence is the primary subject of the article.

If AI could be removed and the article would still make sense,
then it is NOT relevant.

========================

ALWAYS RELEVANT

✓ New AI models
✓ LLM releases
✓ AI research
✓ AI benchmarks
✓ AI safety
✓ AI regulation
✓ AI policy
✓ AI infrastructure
✓ AI chips specifically because of AI
✓ AI developer tools
✓ AI agents
✓ AI APIs
✓ Open-source AI models
✓ Major AI product launches
✓ AI training methods
✓ AI reasoning
✓ AI robotics where AI is central

========================

NOT RELEVANT

✗ Executive hiring
✗ CEO interviews
✗ Company restructuring
✗ Funding rounds
✗ Venture capital
✗ Acquisitions
✗ Stock market news
✗ Earnings reports
✗ General business news
✗ Product management
✗ Health stories using AI as a tool
✗ Vision Pro leadership changes
✗ Uber executive moves
✗ Generic startup news

Unless the article itself is fundamentally about AI technology.

========================

SPECIAL RULES

If the article is about:
OpenAI, Anthropic, Google DeepMind, xAI, Meta AI

DO NOT automatically mark it relevant.
Company news is relevant ONLY when the actual topic is AI technology,
AI models, AI research, AI deployment, or AI policy.

========================

CONFIDENCE CALIBRATION (CRITICAL):

- 0.90-1.00: AI is unambiguously the main subject. No doubt.
- 0.70-0.89: AI is likely the main subject but there is minor ambiguity.
- 0.50-0.69: Borderline. AI is present but may not be central.
- Below 0.50: AI is incidental or not the main subject.

Do NOT default to 1.00. Use the full range. Be honest about uncertainty.

========================

HARD CONSTRAINTS

- Do not infer missing facts.
- Do not speculate.
- Base every decision only on the supplied article.
- Do not classify based on company name alone.
- Do not add external knowledge.

Return ONLY a structured JSON object."""

USER_PROMPT_TEMPLATE = """Classify the following article:

Title: {title}
Content: {content}

Return JSON with:

{{
  "relevant": boolean,
  "confidence": float (0 to 1),
  "reason": "one short sentence explicitly referencing the decision rule"
}}

Reason examples:
- "The article focuses on executive hiring rather than AI technology."
- "The primary subject is an AI model release."
- "The article is about AI regulation."
- "The article discusses AI chips as core infrastructure."
"""


async def classify_article(
    article: Article,
    article_id: int,
    api_base: str,
    api_key: str,
    model: str = "mistral-medium-3-5",
    proxy: str | None = None,
) -> ClassificationResult:
    """Classify a single article using an LLM. Safe wrapper — never raises."""

    try:
        user_prompt = USER_PROMPT_TEMPLATE.format(
            title=article.title,
            content=article.content[:2000],
        )

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 200,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        MAX_RETRIES = 3
        RETRY_DELAY = 2

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
                    raw = data["choices"][0]["message"]["content"].strip()

                    if raw.startswith("```"):
                        raw = raw.split("```")[1]
                        if raw.startswith("json"):
                            raw = raw[4:]
                        raw = raw.strip()

                    try:
                        result = json.loads(raw)
                    except json.JSONDecodeError:
                        result = {"relevant": False, "confidence": 0.0, "reason": f"Parse error: {raw[:100]}"}

                    return ClassificationResult(
                        article_id=article_id,
                        relevant=result.get("relevant", False),
                        confidence=float(result.get("confidence", 0.0)),
                        reason=result.get("reason", "No reason provided"),
                        article_title=article.title,
                        article_url=article.url,
                    )
            except httpx.HTTPError as e:
                if attempt < MAX_RETRIES - 1:
                    import asyncio
                    await asyncio.sleep(RETRY_DELAY)
                    continue
                raise

        raise RuntimeError("All retries failed")

    except Exception as e:
        return ClassificationResult(
            article_id=article_id,
            relevant=False,
            confidence=0.0,
            reason="CLASSIFICATION_FAILED",
            article_title=article.title,
            article_url=article.url,
            error=str(e),
        )
