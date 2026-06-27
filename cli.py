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

from classifier.fetcher import fetch_rss, fetch_url, fetch_multiple_urls
from classifier.classifier import classify_article
from classifier.processor import process as phase2_process, ProcessedItem
from classifier.models import RunSummary, Article, ClassificationResult

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
    process: bool = typer.Option(False, "--process", "-p", help="Run Phase 2 (dedup + noise removal + normalization)"),
):
    """Classify articles from an RSS feed or URL."""
    config = load_config()

    _api_base = api_base or config.get("api_base", "https://api.openai.com/v1")
    _api_key = api_key or config.get("api_key", "")
    _model = model or config.get("model", "gpt-3.5-turbo")

    if not _api_key:
        console.print("[red]Error: No API key. Set in config.yaml or --api-key[/red]")
        raise typer.Exit(1)

    asyncio.run(_run_classify(source, _api_base, _api_key, _model, output, limit, process))


async def _run_classify(source: str, api_base: str, api_key: str, model: str, output: str | None, limit: int, run_phase2: bool):
    """Async classification pipeline."""
    # Step 1: Fetch articles
    console.print(f"\n[bold blue]📥 Fetching from:[/bold blue] {source}")

    if source.startswith("http") and any(source.endswith(ext) for ext in [".xml", ".rss", ".atom"]) or "feed" in source or "rss" in source:
        articles = await fetch_rss(source, limit=limit)
        feed_type = "RSS"
    elif source.startswith("http"):
        article = await fetch_url(source)
        articles = [article]
        feed_type = "URL"
    else:
        console.print("[red]Error: Invalid source. Provide an RSS feed URL or article URL.[/red]")
        raise typer.Exit(1)

    if not articles:
        console.print("[yellow]No articles found.[/yellow]")
        raise typer.Exit(0)

    console.print(f"[green]✓ Found {len(articles)} article(s)[/green]\n")

    # Step 2: Classify each article
    classifications = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for i, article in enumerate(articles):
            task = progress.add_task(f"Classifying [{i+1}/{len(articles)}]: {article.title[:50]}...", total=None)
            try:
                result = await classify_article(article, api_base, api_key, model)
                classifications.append(result)
            except Exception as e:
                console.print(f"  [red]✗ Error: {e}[/red]")
            progress.update(task, completed=True)

    # Step 3: Display classification results
    _display_classifications(classifications)

    # Step 4: Phase 2 processing (if requested)
    processed_data = None
    if run_phase2:
        console.print("\n[bold magenta]🔄 Phase 2: Processing...[/bold magenta]")

        # Filter for relevant items only
        relevant_articles = []
        relevant_classifications = []
        for art, cls in zip(articles, classifications):
            if cls.relevant:
                relevant_articles.append(art)
                relevant_classifications.append(cls)

        if not relevant_articles:
            console.print("[yellow]No relevant articles to process.[/yellow]")
        else:
            processed_data = phase2_process(relevant_articles, relevant_classifications)
            _display_processed(processed_data)

    # Step 5: Save if requested
    if output:
        _save_results(classifications, output, processed_data)

    # Summary
    relevant_count = sum(1 for r in classifications if r.relevant)
    avg_conf = sum(r.confidence for r in classifications) / len(classifications) if classifications else 0
    console.print(f"\n[bold]📊 Phase 1 Summary:[/bold] {relevant_count}/{len(classifications)} relevant | Avg confidence: {avg_conf:.2f}")

    if processed_data is not None:
        console.print(f"[bold]🧹 Phase 2 Summary:[/bold] {len(processed_data)} items after dedup + noise removal")


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
    table.add_column("Source", max_width=20)

    for i, item in enumerate(processed_data, 1):
        source = (item.get("source") or "")[:20]
        table.add_row(str(i), item["title"][:45], f"{item['confidence']:.2f}", source)

    console.print(table)


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
