"""Phase 4: Telegram publisher — sends generated posts to a channel."""

import asyncio
import re
import httpx


def convert_to_telegram_html(text: str) -> str:
    """Convert LLM output (*bold* format) to Telegram HTML.
    
    Steps:
    1. Escape HTML special chars (&, <, >)
    2. Convert *bold* → <b>bold</b>
    3. Leave everything else as-is (bullets, emojis, URLs)
    """
    # Step 1: Escape HTML
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")

    # Step 2: *bold* → <b>bold</b> (non-greedy, single line)
    text = re.sub(r'\*([^*\n]+)\*', r'<b>\1</b>', text)

    return text


def _split_message(text: str, max_len: int = 4096) -> list[str]:
    """Split a long message into chunks respecting Telegram's 4096 char limit.
    
    Splits at double newline boundaries when possible.
    """
    if len(text) <= max_len:
        return [text]
    
    chunks = []
    while len(text) > max_len:
        split_point = text[:max_len].rfind("\n\n")
        if split_point == -1:
            split_point = max_len
        chunks.append(text[:split_point])
        text = text[split_point:].strip()
    if text:
        chunks.append(text)
    return chunks


async def send_post(
    text: str,
    token: str,
    channel: str,
    proxy: str | None = None,
) -> bool:
    """Send a single post to Telegram channel.
    
    Converts *bold* Markdown to HTML before sending.
    Returns True if successful, False otherwise.
    """
    url = f"https://api.telegram.org/bot{token}/sendMessage"

    # LLM already outputs Telegram HTML — no conversion needed.
    # Do NOT escape HTML here; it would destroy the tags.

    payload = {
        "chat_id": channel,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": False,
    }
    
    try:
        async with httpx.AsyncClient(
            timeout=30,
            proxy=proxy,
            verify=False,
        ) as client:
            # Telegram limit: 4096 chars. Split if needed.
            chunks = _split_message(text, max_len=4096)
            for chunk in chunks:
                payload["text"] = chunk
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
            return True
    except Exception as e:
        print(f"  ✗ Telegram error: {e}")
        return False


async def publish_posts(
    posts: list[str],
    token: str,
    channel: str,
    proxy: str | None = None,
    delay_seconds: int = 3,
) -> dict:
    """Send all posts to Telegram channel with delay between each.
    
    delay_seconds: فاصله بین پستها تا spam نشود
    
    Returns summary: {"sent": int, "failed": int}
    """
    sent = 0
    failed = 0

    for i, post in enumerate(posts):
        print(f"  Sending post {i+1}/{len(posts)}...")
        success = await send_post(post, token, channel, proxy)
        
        if success:
            sent += 1
        else:
            failed += 1
        
        # تأخیر بین پستها
        if i < len(posts) - 1:
            await asyncio.sleep(delay_seconds)

    return {"sent": sent, "failed": failed}
