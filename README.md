# AI News Classifier

A 4-phase CLI pipeline that fetches news from RSS feeds, classifies articles by AI relevance using LLM, cleans and deduplicates results, generates bilingual Telegram posts, and publishes them to a Telegram channel.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Pipeline Phases](#pipeline-phases)
  - [Phase 1: Fetch & Classify](#phase-1-fetch--classify)
  - [Phase 2: Process & Deduplicate](#phase-2-process--deduplicate)
  - [Phase 3: Generate Posts](#phase-3-generate-posts)
  - [Phase 4: Publish to Telegram](#phase-4-publish-to-telegram)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Setup](#setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Prompt Engineering](#prompt-engineering)
- [Design Decisions](#design-decisions)

---

## Overview

This tool automates the process of monitoring tech news for AI-related content. It's designed to:

1. **Fetch** articles from RSS feeds (TechCrunch, The Verge, etc.)
2. **Classify** each article using an LLM to determine AI relevance with calibrated confidence scores
3. **Process** results — deduplicate similar stories, remove noise, normalize metadata
4. **Generate** bilingual (Persian + English) Telegram posts with grounded summaries
5. **Publish** posts to a Telegram channel via Bot API

The system works with any **OpenAI-compatible API** (OpenRouter, local llama.cpp, vLLM, etc.) and supports HTTP/SOCKS proxies for restricted networks.

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  RSS Feeds   │────▶│   Phase 1    │────▶│   Phase 2    │────▶│   Phase 3    │────▶│   Phase 4    │
│  (feedparser)│     │  Classifier  │     │  Processor   │     │  Generator   │     │  Publisher   │
│              │     │  (LLM API)   │     │  (rule-based)│     │  (LLM API)   │     │  (Bot API)   │
└─────────────┘     └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                           │                     │                     │                     │
                     Relevant/Not          Deduplicated          Persian +            Telegram
                     + Confidence          + Cleaned             English              Channel
                     + Reason              + Normalized          Posts                (@channel)
```

### Data Flow

```
RSS URL → fetcher.py → list[Article]
    ↓
Article[] → classifier.py → list[Classification]  (each with: relevant, confidence, reason, summary, facts)
    ↓
filter relevant only
    ↓
Article[] + Classification[] → processor.py → list[ProcessedItem]  (deduped, noise-removed)
    ↓
ProcessedItem[] → generator.py → list[str]  (Telegram-ready posts)
    ↓
str[] → publisher.py → Telegram Bot API → Channel
```

---

## Pipeline Phases

### Phase 1: Fetch & Classify

**Module:** `classifier/fetcher.py` + `classifier/classifier.py`

- Parses RSS feeds using `feedparser` or fetches a single URL
- Each article (title + body + source) is sent to an LLM with a structured prompt
- LLM returns a JSON object with:
  - `relevant`: boolean — is this about AI/ML/LLMs?
  - `confidence`: float 0.0–1.0 — how confident is the classification?
  - `reason`: one-sentence explanation
  - `summary`: 1–2 sentence grounded summary (only from article text)
  - `facts`: list of factual bullet points extracted from the article
- Uses **2-attempt retry** on parse failures
- Returns `ClassificationResult` Pydantic model per article

**Confidence calibration:** The prompt includes explicit tier guidance to prevent the LLM from defaulting to `1.00`:
- `0.95–1.00` = clearly AI-focused
- `0.80–0.94` = AI mentioned prominently
- `0.60–0.79` = AI is secondary topic
- `0.40–0.59` = tangential AI mention
- `0.00–0.39` = barely related

### Phase 2: Process & Deduplicate

**Module:** `classifier/processor.py`

- Rule-based (no LLM calls) — fast and deterministic
- **Title normalization:** strips source prefixes like "TechCrunch:", "The Verge |", date suffixes, clickbait phrases
- **Deduplication:** groups articles with >60% word overlap in titles, keeps highest confidence
- **Noise removal:** filters out items with confidence < 0.7, empty summaries, or too-short titles
- **Fact flattening:** joins list of facts into a single string for downstream use

### Phase 3: Generate Posts

**Module:** `classifier/generator.py`

- Takes processed items and generates bilingual Telegram posts via LLM
- Each post includes:
  - Persian headline with 🔴 or 🟡 emoji (based on confidence)
  - Grounded summary (3–4 sentences, Persian)
  - "Why it matters" section (1–2 sentences explaining real-world impact)
  - English tags: `#Topic` + `#Source`
  - Source link
- **Batch processing:** 2 items per LLM call (avoids prompt overflow)
- **Fallback:** if batch fails, retries each item individually
- Posts are returned as plain strings, ready for publishing

### Phase 4: Publish to Telegram

**Module:** `classifier/publisher.py`

- Sends posts to a Telegram channel via Bot API (`sendMessage`)
- Uses `HTML` parse mode (not Markdown) for reliable formatting
- Handles Telegram's 4096-char limit by splitting at paragraph boundaries
- 3-second delay between posts to avoid rate limits
- Supports `--dry-run` to preview without sending

---

## Project Structure

```
ai-news-classifier/
├── cli.py                    # Typer CLI entry point — orchestrates all 4 phases
├── config.yaml               # Runtime config (API keys, feeds, Telegram token)
├── config.example.yaml       # Template config (safe to commit)
├── requirements.txt          # Python dependencies
├── .gitignore
├── classifier/
│   ├── __init__.py
│   ├── fetcher.py            # Phase 1a: RSS/URL fetching (feedparser + httpx)
│   ├── classifier.py         # Phase 1b: LLM classification (OpenAI-compatible API)
│   ├── models.py             # Pydantic models (Article, ClassificationResult)
│   ├── processor.py          # Phase 2: Rule-based dedup + noise removal
│   ├── generator.py          # Phase 3: LLM post generation (Persian + English)
│   └── publisher.py          # Phase 4: Telegram Bot API publisher
└── *.json / *.txt            # Test results and outputs (not committed)
```

---

## Tech Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| CLI framework | `typer` + `rich` | Command-line interface with colored output, tables, progress bars |
| HTTP client | `httpx` (async) | Async HTTP with proxy support (SOCKS/HTTP) |
| RSS parsing | `feedparser` | Parse RSS/Atom feeds |
| Data models | `pydantic` v2 | Typed models with validation (Article, ClassificationResult) |
| Config | `pyyaml` | YAML config file loading |
| LLM API | OpenAI-compatible | Any `/v1/chat/completions` endpoint (OpenRouter, local, etc.) |
| Telegram | Bot API (httpx) | Direct HTTP calls to `api.telegram.org` |

---

## Setup

### Prerequisites

- Python 3.12+
- An OpenAI-compatible API key (e.g., OpenRouter, Mistral, local llama.cpp)
- (Optional) Telegram Bot token for publishing

### Installation

```bash
git clone git@github.com:sajjadele/ai-news-classifier.git
cd ai-news-classifier
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configuration

```bash
cp config.example.yaml config.yaml
# Edit config.yaml with your API key, model, and optional Telegram credentials
```

**config.yaml fields:**

```yaml
api_base: "https://router.bynara.id/v1"   # OpenAI-compatible endpoint
api_key: "YOUR_API_KEY"                    # API key
model: "mistral-medium-3-5"               # Model name
proxy: "http://127.0.0.1:10808/"          # HTTP/SOCKS proxy (optional)

feeds:                                      # RSS feeds to monitor
  - name: "TechCrunch AI"
    url: "https://techcrunch.com/category/artificial-intelligence/feed/"

telegram_token: "YOUR_BOT_TOKEN"           # For Phase 4 (optional)
telegram_channel: "@your_channel"          # Target channel
```

---

## Usage

### Full Pipeline (all 4 phases)

```bash
python3 cli.py classify "https://techcrunch.com/category/artificial-intelligence/feed/" \
  --limit 10 \
  --process \
  --generate \
  --posts-file posts.txt \
  --send
```

### Classify Only (Phase 1)

```bash
python3 cli.py classify "https://techcrunch.com/category/artificial-intelligence/feed/" \
  --limit 10 \
  --output results.json
```

### Classify + Process (Phase 1 + 2)

```bash
python3 cli.py classify RSS_URL --limit 10 --process --output results.json
```

### Dry Run (preview posts without sending)

```bash
python3 cli.py classify RSS_URL --limit 10 -p -g --dry-run
```

### Single Article

```bash
python3 cli.py classify "https://example.com/article-url"
```

### CLI Flags

| Flag | Short | Description |
|------|-------|-------------|
| `--output` | `-o` | Save results to JSON file |
| `--limit` | `-n` | Max articles from RSS feed (default: 10) |
| `--model` | `-m` | Override LLM model from config |
| `--api-base` | | Override API base URL |
| `--api-key` | | Override API key |
| `--process` | `-p` | Run Phase 2 (dedup + noise removal) |
| `--generate` | `-g` | Run Phase 3 (Telegram post generation) |
| `--posts-file` | | Save generated posts to .txt file |
| `--publish` | | Send posts to Telegram |
| `--send` | `-s` | Alias for `--publish` |
| `--dry-run` | | Show posts without sending (mutually exclusive with `--send`) |

---

## Prompt Engineering

The classifier prompt is the core of Phase 1 accuracy. Key design choices:

### Two-Message Structure
- **System message:** defines the AI's role and task
- **User message:** contains the article data (title, body, source)

This separation improves reliability compared to a single combined prompt.

### Confidence Calibration
Without explicit guidance, LLMs default to `confidence: 1.00` for everything. The prompt includes a tiered scale:
```
0.95–1.00 = clearly AI-focused
0.80–0.94 = AI mentioned prominently
0.60–0.79 = AI is secondary topic
0.40–0.59 = tangential AI mention
0.00–0.39 = barely related
```

### Grounded Summaries
The prompt explicitly forbids hallucination:
> "Summarize using ONLY the information in the article. Do NOT add external knowledge."

### Strict Relevance Criteria
Articles must focus on **applied AI** — not just mention "intelligence" or "smart" in other contexts. Examples:
- ✅ "OpenAI launches new model" → clearly AI
- ✅ "AI-powered drug discovery startup raises $50M" → applied AI
- ❌ "The intelligent design of modern buildings" → not AI
- ❌ "Smart home market grows 15%" → IoT, not AI/ML

---

## Design Decisions

### Why Rule-Based Phase 2?
Deduplication and noise removal are deterministic tasks. Using an LLM would add latency and cost without improving accuracy. The rule-based approach is:
- **Fast:** no API calls
- **Predictable:** same input → same output
- **Debuggable:** easy to trace why something was removed

### Why HTML Parse Mode for Telegram?
Markdown formatting in Telegram is unreliable (underscores in URLs break italic, etc.). HTML with `<b>bold</b>` tags is more predictable.

### Why Batch Size 2 for Post Generation?
Each processed item needs a headline, summary, "why it matters", and tags. With 2 items per call, the prompt stays under most model context limits while reducing API calls by 50% compared to item-per-call.

### Why Proxy Support?
The project is developed in Iran, where direct access to external APIs requires a proxy. All HTTP clients (`httpx`) accept an optional `proxy` parameter, and `verify=False` is used for self-signed certificates common in proxy setups.

---

## Example Output

### Phase 1 Classification Table
```
┌───┬──────────────────────────────────┬────────────┬────────────┬──────────────────────────────┐
│ # │ Title                            │ Relevant   │ Confidence │ Reason                       │
├───┼──────────────────────────────────┼────────────┼────────────┼──────────────────────────────┤
│ 1 │ OpenAI releases GPT-5            │ ✓ Yes      │ 0.98       │ Direct AI model release      │
│ 2 │ Tech stocks rise amid optimism   │ ✗ No       │ 0.15       │ General market news          │
│ 3 │ AI drug discovery startup        │ ✓ Yes      │ 0.92       │ Applied AI in healthcare     │
└───┴──────────────────────────────────┴────────────┴────────────┴──────────────────────────────┘
```

### Phase 3 Generated Post
```
🔴 خبر فوری هوش مصنوعی

*اپن‌ای‌ای مدل GPT-5 را معرفی کرد*

اپن‌ای‌ای امروز نسل جدید مدل زبانی خود، GPT-5، را منتشر کرد.
این مدل قابلیتهای پیشرفته‌تری در استدلال و تحلیل تصویر دارد.

⚡ چرا مهمه؟
رقابت بین شرکتهای هوش مصنوعی با سرعت بالایی ادامه دارد و این نسخه جدید استانداردهای صنعت را بالاتر می‌برد.

#GPT5 #OpenAI

🔗 TechCrunch
```

---

## License

This project is for personal/educational use.
