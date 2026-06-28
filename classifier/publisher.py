"""Phase 4: Telegram publisher — sends generated posts to a channel."""

import asyncio
import httpx


async def send_post(
    text: str,
    token: str,
    channel: str,
    proxy: str | None = None,
) -> bool:
    """Send a single post to Telegram channel.
    
    Returns True if successful, False otherwise.
    """
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    
    payload = {
        "chat_id": channel,
        "text": text,
        "parse_mode": "Markdown",
    }
    
    try:
        async with httpx.AsyncClient(
            timeout=30,
            proxy=proxy,
            verify=False,
        ) as client:
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
