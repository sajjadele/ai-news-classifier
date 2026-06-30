# AI News Classifier

A four-phase AI news pipeline that fetches articles from RSS feeds or URLs, classifies AI relevance with an LLM, removes duplicates/noise, generates Persian Telegram posts, and publishes them to a Telegram channel.

## Quick Start

```bash
git clone git@github.com:sajjadele/ai-news-classifier.git
cd ai-news-classifier

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

cp config.example.yaml config.yaml
# Fill in your API key and optional Telegram settings

python3 cli.py classify "https://techcrunch.com/category/artificial-intelligence/feed/" \
  --limit 5 \
  --process \
  --generate \
  --dry-run
```

To publish generated posts to Telegram:

```bash
python3 cli.py classify RSS_URL --process --generate --send
```

---

## Overview

The pipeline is designed for monitoring AI news and converting relevant stories into Telegram-ready posts.

Pipeline stages:

1. Fetch articles from RSS feeds or direct URLs
2. Classify whether AI is the primary subject of the article
3. Remove duplicates and obvious noise
4. Generate Persian Telegram posts with HTML formatting
5. Publish posts through the Telegram Bot API

The project works with any OpenAI-compatible API endpoint.

---

## Architecture

```
RSS Feed / URL
       ↓
fetcher.py
       ↓
classifier.py
       ↓
processor.py
       ↓
generator.py
       ↓
publisher.py
       ↓
Telegram Channel
```

---

## Pipeline Phases

### Phase 1: Fetch & Classify

Modules:
- `classifier/fetcher.py`
- `classifier/classifier.py`

Features:
- Fetches RSS feeds using `feedparser`
- Supports direct article URLs
- Uses async `httpx` clients
- Removes basic HTML tags before classification
- Limits article content length before sending to the LLM
- Returns structured `ClassificationResult` objects

Classification behavior:
- The classifier only marks articles as relevant when AI is the primary subject
- Company names alone are not enough for relevance
- The prompt explicitly distinguishes AI technology news from business/newsroom activity

Actual confidence calibration from `SYSTEM_PROMPT`:

- `0.90–1.00` → AI is unambiguously the main subject
- `0.70–0.89` → AI is likely the main subject with minor ambiguity
- `0.50–0.69` → borderline relevance
- `<0.50` → AI is incidental or not central

The classifier uses:
- temperature `0.0`
- max tokens `200`
- up to `3` retries for HTTP/API failures
- JSON parsing with fallback handling when the model returns invalid output

Returned fields:

```json
{
  "relevant": true,
  "confidence": 0.91,
  "reason": "The primary subject is an AI model release."
}
```

### Phase 2: Process & Deduplicate

Module:
- `classifier/processor.py`

This phase is fully rule-based and deterministic.

Processing steps:

1. Deduplication
2. Noise removal
3. Title normalization

Deduplication method:
- Uses `difflib.SequenceMatcher`
- Duplicate threshold: `0.82` title similarity
- Requires publication times within a `48 hour` window
- Keeps the higher-confidence item
- Uses content length as a tie-breaker

Noise removal rules:
- Removes empty content
- Removes content shorter than `80` characters
- Removes promotional/newsletter-style articles

Title normalization:
- Removes excessive emoji usage
- Normalizes whitespace
- Preserves article meaning

### Phase 3: Generate Posts

Module:
- `classifier/generator.py`

Generation behavior:
- Each article is processed individually
- No batching is used
- Posts with confidence below `0.70` are skipped
- Uses HTML-formatted Telegram output
- Uses one LLM call per post

Formatting rules enforced in the prompt:
- `<b>` for titles
- `<blockquote>` for bullet summaries
- `<i>` for analysis text
- `<a href="...">` for source links
- Markdown formatting is explicitly forbidden

Hashtag rules from the prompt:
- All hashtags appear on one line
- Maximum of 4 hashtags total
- One required category hashtag:
  - `#پژوهش`
  - `#سیاست`
  - `#صنعت`
  - `#محصول`
- Additional entity/topic hashtags are added when relevant

The generator uses:
- temperature `0.5`
- max tokens `600`
- up to `3` retries for HTTP/API failures

### Phase 4: Publish to Telegram

Module:
- `classifier/publisher.py`

Publishing behavior:
- Sends messages through Telegram Bot API `sendMessage`
- Uses `parse_mode="HTML"`
- Uses `disable_web_page_preview=False`
- Splits messages larger than Telegram's 4096-character limit
- Splits at double-newline boundaries when possible
- Waits `3` seconds between posts by default

Retry/error behavior:
- Publisher does not retry failed Telegram sends
- Errors are caught and logged
- `send_post()` returns `True` or `False`
- `publish_posts()` tracks sent vs failed posts

---

## Project Structure

```text
ai-news-classifier/
├── cli.py
├── config.yaml
├── config.example.yaml
├── requirements.txt
├── classifier/
│   ├── fetcher.py
│   ├── classifier.py
│   ├── generator.py
│   ├── models.py
│   ├── processor.py
│   └── publisher.py
└── eval/
    ├── run_eval.py
    └── test_set.csv
```

---

## Configuration

Create your local configuration:

```bash
cp config.example.yaml config.yaml
```

Example configuration:

```yaml
api_base: "https://your-openai-compatible-api/v1"
api_key: "YOUR_API_KEY"
model: "your-model-name"

proxy: "http://YOUR_PROXY:PORT"

feeds:
  - name: "TechCrunch AI"
    url: "https://techcrunch.com/category/artificial-intelligence/feed/"

telegram_token: "YOUR_BOT_TOKEN"
telegram_channel: "@your_channel"
```

All runtime values are loaded from `config.yaml`.

---

## Usage

Full pipeline:

```bash
python3 cli.py classify RSS_URL \
  --limit 10 \
  --process \
  --generate \
  --send
```

Classification only:

```bash
python3 cli.py classify RSS_URL --limit 10
```

Generate posts without publishing:

```bash
python3 cli.py classify RSS_URL \
  --process \
  --generate \
  --dry-run
```

Single article URL:

```bash
python3 cli.py classify "https://example.com/article"
```

---

## Example Output

Sample generated posts from the current pipeline:

```text
🧠 پایان انحصار Nvidia در تراشه‌های هوش مصنوعی (Custom AI Chips Shift)

• OpenAI تراشه سفارشی Jalapeño (با همکاری Broadcom) را برای استنتاج معرفی کرد
• شرکت‌هایی مانند Google، Apple و SpaceX نیز در حال توسعه تراشه‌های اختصاصی برای کاهش وابستگی به یک تامین‌کننده هستند
• این حرکت خطر تک‌تامینی (single-supplier risk) را در زنجیره تامین AI کاهش می‌دهد

🔍 چرا مهم است:
Technical shift — وابستگی مطلق به Nvidia در حال جایگزیری با معماری‌های سفارشی و متنوع‌تر است

#AIHardware
📖 مطالعه کامل مقاله در TechCrunch
https://techcrunch.com/video/why-everyone-from-openai-to-spacex-is-building-their-own-chips-and-turning-up-the-heat-on-nvidia/
```

---

## Tech Stack

- `httpx` — async HTTP client
- `feedparser` — RSS parsing
- `pydantic` — typed data models
- `pyyaml` — YAML config loading
- `typer` + `rich` — CLI interface

---

## License

Personal and educational use.
