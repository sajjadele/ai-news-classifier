"""AI News Classifier — CLI entry point."""

import asyncio
import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from classifier.fetcher import fetch_rss, fetch_url
from classifier.classifier import classify_article
from classifier.processor import process as phase2_process
from classifier.generator import generate_telegram_posts, save_posts
from classifier.publisher import publish_posts

app = typer.Typer(help="AI News Classifier — filter AI-relevant articles from news feeds")
console = Console()


def load_config() -> dict:
    """Load config from config.yaml."""
    import yaml
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


@app.command()
def classify(
    source: str = typer.Argument(help="RSS feed URL or article URL"),
    output: str = typer.Option(None, "--output", "-o", help="Save results to JSON file"),
    limit: int = typer.Option(10, "--limit", "-n", help="Max articles from RSS feed"),
    model: str = typer.Option(None, "--model", "-m", help="Override LLM model"),
    api_base: str = typer.Option(None, "--api-base", help="API base URL"),
    api_key: str = typer.Option(None, "--api-key", help="API key"),
    process: bool = typer.Option(False, "--process", "-p", help="Run Phase 2 (dedup + noise removal)"),
    generate: bool = typer.Option(False, "--generate", "-g", help="Run Phase 3 (Telegram post generation)"),
    posts_file: str = typer.Option(None, "--posts-file", help="Save Telegram posts to .txt file"),
    publish: bool = typer.Option(False, "--publish", help="Send posts to Telegram"),
    send: bool = typer.Option(False, "--send", "-s", help="Send posts to Telegram (alias for --publish)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show posts without sending"),
):
    """Classify articles from an RSS feed or URL."""
    config = load_config()

    _api_key = api_key or config.get("api_key", "")
    if not _api_key:
        console.print("[red]Error: No API key. Set in config.yaml or --api-key[/red]")
        raise typer.Exit(1)

    _api_base = api_base or config.get("api_base", "https://api.openai.com/v1")
    _model = model or config.get("model", "gpt-3.5-turbo")
    _proxy = config.get("proxy", None)

    will_send = publish or send
    if dry_run and will_send:
        console.print("[red]Error: Cannot use --dry-run with --send or --publish. "
                      "These flags are mutually exclusive.[/red]")
        raise typer.Exit(1)

    if dry_run:
        console.print("[yellow]ℹ Dry-run mode: posts will be shown but NOT sent.[/yellow]")

    asyncio.run(_run_pipeline(source, _api_base, _api_key, _model, _proxy, output, limit, process, generate, posts_file, will_send, dry_run, config))


async def _run_pipeline(
    source: str, api_base: str, api_key: str, model: str, proxy: str | None,
    output: str | None, limit: int,
    run_phase2: bool, run_phase3: bool, posts_file: str | None,
    publish: bool = False, dry_run: bool = False,
    config: dict | None = None,
):
    """Full pipeline: Phase 1 → Phase 2 → Phase 3."""

    # ── Phase 1: Fetch + Classify ──────────────────────────────────────
    console.print(f"\n[bold blue]📥 Phase 1: Fetching from:[/bold blue] {source}")

    if "feed" in source or "rss" in source or source.endswith((".xml", ".rss", ".atom")):
        articles = await fetch_rss(source, limit=limit, proxy=proxy)
    elif source.startswith("http"):
        article = await fetch_url(source, proxy=proxy)
        articles = [article]
    else:
        console.print("[red]Error: Invalid source. Provide an RSS feed URL or article URL.[/red]")
        raise typer.Exit(1)

    if not articles:
        console.print("[yellow]No articles found.[/yellow]")
        raise typer.Exit(0)

    console.print(f"[green]✓ Found {len(articles)} article(s)[/green]\n")

    classifications = []
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        for i, article in enumerate(articles):
            task = progress.add_task(f"Classifying [{i+1}/{len(articles)}]: {article.title[:50]}...", total=None)
            result = await classify_article(article, i, api_base, api_key, model, proxy=proxy)
            classifications.append(result)
            if result.error:
                console.print(f"  [red]✗ Error: {result.error}[/red]")
            progress.update(task, completed=True)

    _display_classifications(classifications)

    relevant_count = sum(1 for r in classifications if r.relevant)
    failed_count = sum(1 for c in classifications if c.error)
    avg_conf = sum(r.confidence for r in classifications) / len(classifications) if classifications else 0
    if failed_count == len(classifications):
        console.print(f"[red]⚠ All {failed_count} classification(s) failed. Check API key and endpoint.[/red]")
    elif failed_count > 0:
        console.print(f"[yellow]⚠ {failed_count} classification(s) failed. Results may be incomplete.[/yellow]")
    console.print(f"\n[bold]📊 Phase 1:[/bold] {relevant_count}/{len(articles)} relevant | {len(classifications)} classified | {failed_count} failed | Avg confidence: {avg_conf:.2f}")

    # ── Phase 2: Processing ────────────────────────────────────────────
    processed_data = None
    if run_phase2:
        console.print("\n[bold magenta]🔄 Phase 2: Processing...[/bold magenta]")

        assert len(classifications) == len(articles), \
            f"Phase 1 broken: {len(classifications)} classifications for {len(articles)} articles"
        assert all(c.article_id is not None for c in classifications), \
            "Phase 1 broken: classification missing article_id"

        relevant_articles = []
        relevant_classifications = []
        for art, cls in zip(articles, classifications):
            if cls.relevant and cls.error is None:
                relevant_articles.append(art)
                relevant_classifications.append(cls)

        assert all(cls.relevant and cls.error is None for cls in relevant_classifications), \
            "No rejected article may enter Phase 2."

        if not relevant_articles:
            console.print("[yellow]No relevant articles to process.[/yellow]")
        else:
            processed_data = phase2_process(relevant_articles, relevant_classifications)
            _display_processed(processed_data)
            console.print(f"[bold]🧹 Phase 2:[/bold] {len(processed_data)} items after dedup + noise removal")

    # ── Phase 3: Telegram Post Generation ──────────────────────────────
    posts = None
    if run_phase3:
        if processed_data is None:
            if run_phase2:
                console.print("[red]Error: No relevant articles to generate posts from.[/red]")
            else:
                console.print("[red]Error: --generate requires --process. Use both flags.[/red]")
            raise typer.Exit(1)

        console.print("\n[bold cyan]📝 Phase 3: Generating Telegram posts...[/bold cyan]")

        assert all(item.get("relevant", False) for item in processed_data), \
            "Every generated post must correspond to a Relevant == YES article."

        posts = await generate_telegram_posts(processed_data, api_base, api_key, model, proxy=proxy)
        _display_posts_preview(posts)
        console.print(f"[bold]📝 Phase 3:[/bold] {len(posts)} posts generated")

        # Save posts
        if posts_file:
            save_posts(posts, posts_file)
            console.print(f"[green]💾 Posts saved to: {posts_file}[/green]")

# ── Phase 4: Publish to Telegram ──────────────────────────────
    if posts and (publish or dry_run):
        _telegram_token = (config or {}).get("telegram_token", "")
        _telegram_channel = (config or {}).get("telegram_channel", "")
        
        if not _telegram_token or not _telegram_channel:
            console.print("[red]Error: telegram_token and telegram_channel must be set in config.yaml[/red]")
        elif dry_run:
            console.print("\n[bold yellow]🔍 Dry Run — پست‌ها ارسال نمی‌شوند:[/bold yellow]")
            for i, post in enumerate(posts, 1):
                console.print(f"\n[dim]─── Post {i} ───[/dim]")
                console.print(post)
        else:
            console.print("\n[bold green]📤 Phase 4: Sending to Telegram...[/bold green]")
            result = await publish_posts(
                posts=posts,
                token=_telegram_token,
                channel=_telegram_channel,
                proxy=proxy,
            )
            console.print(f"[bold]📤 Phase 4:[/bold] {result['sent']} sent, {result['failed']} failed")
    # ── Save JSON ──────────────────────────────────────────────────────
    if output:
        _save_results(classifications, output, processed_data)

    # ── Final Summary ──────────────────────────────────────────────────
    console.print("\n[bold green]═══ Pipeline Complete ═══[/bold green]")
    console.print(f"  Phase 1: {relevant_count}/{len(articles)} relevant articles ({len(classifications)} classified)")
    if processed_data:
        console.print(f"  Phase 2: {len(processed_data)} cleaned items")
    if posts:
        console.print(f"  Phase 3: {len(posts)} Telegram posts ready")


# ── Display Functions ──────────────────────────────────────────────────────

def _display_classifications(results):
    """Show classification results in a table."""
    table = Table(title="Phase 1: Classification Results", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Title", max_width=40)
    table.add_column("Relevant", justify="center", width=10)
    table.add_column("Confidence", justify="center", width=10)
    table.add_column("Reason", max_width=40)

    for i, r in enumerate(results, 1):
        rel_str = "[green]✓ Yes[/green]" if r.relevant else "[red]✗ No[/red]"
        conf_str = f"{r.confidence:.2f}"
        table.add_row(str(i), r.article_title[:40], rel_str, conf_str, r.reason[:40])

    console.print(table)


def _display_processed(processed_data: list[dict]):
    """Show Phase 2 processed results in a table."""
    table = Table(title="Phase 2: Cleaned & Deduplicated", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Title", max_width=45)
    table.add_column("Confidence", justify="center", width=10)

    for i, item in enumerate(processed_data, 1):
        table.add_row(str(i), item["title"][:45], f"{item['confidence']:.2f}")

    console.print(table)


def _display_posts_preview(posts: list[str]):
    """Show a preview of generated posts."""
    console.print("\n[bold]📋 Posts Preview:[/bold]")
    for i, post in enumerate(posts, 1):
        # Show first 5 lines of each post
        lines = post.split("\n")[:5]
        preview = "\n".join(lines)
        console.print(f"\n[dim]─── Post {i} ───[/dim]")
        console.print(preview)
        if len(post.split("\n")) > 5:
            console.print("[dim]  ...[/dim]")


def _save_results(classifications, output_path, processed_data=None):
    """Save results to JSON file."""
    output = {
        "phase1": {
            "total": len(classifications),
            "relevant": sum(1 for r in classifications if r.relevant),
            "not_relevant": sum(1 for r in classifications if not r.relevant),
            "avg_confidence": sum(r.confidence for r in classifications) / len(classifications) if classifications else 0,
            "results": [r.model_dump() for r in classifications],
        },
    }

    if processed_data is not None:
        output["phase2"] = {
            "total": len(processed_data),
            "items": processed_data,
        }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    console.print(f"\n[green]💾 Results saved to: {output_path}[/green]")


if __name__ == "__main__":
    app()
