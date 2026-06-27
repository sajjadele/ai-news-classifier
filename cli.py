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
):
    """Classify articles from an RSS feed or URL."""
    config = load_config()

    _api_base = api_base or config.get("api_base", "https://api.openai.com/v1")
    _api_key = api_key or config.get("api_key", "")
    _model = model or config.get("model", "gpt-3.5-turbo")

    if not _api_key:
        console.print("[red]Error: No API key. Set in config.yaml or --api-key[/red]")
        raise typer.Exit(1)

    asyncio.run(_run_pipeline(source, _api_base, _api_key, _model, output, limit, process, generate, posts_file))


async def _run_pipeline(
    source: str, api_base: str, api_key: str, model: str,
    output: str | None, limit: int,
    run_phase2: bool, run_phase3: bool, posts_file: str | None,
):
    """Full pipeline: Phase 1 → Phase 2 → Phase 3."""

    # ── Phase 1: Fetch + Classify ──────────────────────────────────────
    console.print(f"\n[bold blue]📥 Phase 1: Fetching from:[/bold blue] {source}")

    if "feed" in source or "rss" in source or source.endswith((".xml", ".rss", ".atom")):
        articles = await fetch_rss(source, limit=limit)
    elif source.startswith("http"):
        article = await fetch_url(source)
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
            try:
                result = await classify_article(article, api_base, api_key, model)
                classifications.append(result)
            except Exception as e:
                console.print(f"  [red]✗ Error: {e}[/red]")
            progress.update(task, completed=True)

    _display_classifications(classifications)

    relevant_count = sum(1 for r in classifications if r.relevant)
    avg_conf = sum(r.confidence for r in classifications) / len(classifications) if classifications else 0
    console.print(f"\n[bold]📊 Phase 1:[/bold] {relevant_count}/{len(classifications)} relevant | Avg confidence: {avg_conf:.2f}")

    # ── Phase 2: Processing ────────────────────────────────────────────
    processed_data = None
    if run_phase2:
        console.print("\n[bold magenta]🔄 Phase 2: Processing...[/bold magenta]")

        relevant_articles = [art for art, cls in zip(articles, classifications) if cls.relevant]
        relevant_classifications = [cls for cls in classifications if cls.relevant]

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
            console.print("[red]Error: --generate requires --process. Use both flags.[/red]")
            raise typer.Exit(1)

        console.print("\n[bold cyan]📝 Phase 3: Generating Telegram posts...[/bold cyan]")

        posts = await generate_telegram_posts(processed_data, api_base, api_key, model)
        _display_posts_preview(posts)
        console.print(f"[bold]📝 Phase 3:[/bold] {len(posts)} posts generated")

        # Save posts
        if posts_file:
            save_posts(posts, posts_file)
            console.print(f"[green]💾 Posts saved to: {posts_file}[/green]")

    # ── Save JSON ──────────────────────────────────────────────────────
    if output:
        _save_results(classifications, output, processed_data)

    # ── Final Summary ──────────────────────────────────────────────────
    console.print("\n[bold green]═══ Pipeline Complete ═══[/bold green]")
    console.print(f"  Phase 1: {relevant_count}/{len(classifications)} relevant articles")
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
