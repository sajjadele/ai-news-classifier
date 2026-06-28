"""LLM-based article classifier — strict relevance classification."""

import json
import httpx
from .models import Article, ClassificationResult


SYSTEM_PROMPT = """You are a strict Relevance Classification Engine.

Your job is to decide whether a news article is primarily about Artificial Intelligence.

Definition of "relevant":
- The main subject must be AI systems, AI models, AI companies, AI infrastructure, AI policy, or AI hardware.
- AI being mentioned as a tool is NOT enough.
- Business hiring, funding, or events are NOT relevant unless AI is the core subject.
- Personnel moves (executives joining/leaving companies) are NOT relevant unless the move is specifically about AI research or AI product leadership.
- General tech news involving AI companies is NOT relevant unless the article is specifically about AI technology, models, or policy.

You MUST NOT:
- Add external knowledge
- Infer beyond the text
- Make assumptions about industry importance
- Drift into summarization or commentary
- Classify based only on company name (OpenAI, Anthropic, etc.) — the article must be about AI itself

CONFIDENCE CALIBRATION (CRITICAL):
- 0.90-1.00: AI is unambiguously the main subject. No doubt.
- 0.70-0.89: AI is likely the main subject but there is minor ambiguity.
- 0.50-0.69: Borderline. AI is present but may not be central.
- Below 0.50: AI is incidental or not the main subject.

Do NOT default to 1.00. Use the full range. Be honest about uncertainty.

Return ONLY a structured JSON object."""

USER_PROMPT_TEMPLATE = """Classify the following article:

Title: {title}
Content: {content}

Return JSON with:

{{
  "relevant": boolean,
  "confidence": float (0 to 1),
  "reason": "short strict justification based only on text"
}}"""


async def classify_article(
    article: Article,
    api_base: str,
    api_key: str,
    model: str = "mistral-medium-3-5",
    proxy: str | None = None,
) -> ClassificationResult:
    """Classify a single article using an LLM."""

    user_prompt = USER_PROMPT_TEMPLATE.format(
        title=article.title,
        content=article.content[:2000],  # Safety cap
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
        except httpx.HTTPStatusError as e:
            if attempt < MAX_RETRIES - 1:
                import asyncio
                await asyncio.sleep(RETRY_DELAY)
                continue
            raise

    # Should never reach here, but just in case
    raise RuntimeError("All retries failed")
