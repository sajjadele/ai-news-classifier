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
from classifier.models import RunSummary

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
):
    """Classify articles from an RSS feed or URL."""
    config = load_config()

    _api_base = api_base or config.get("api_base", "https://api.openai.com/v1")
    _api_key = api_key or config.get("api_key", "")
    _model = model or config.get("model", "gpt-3.5-turbo")

    if not _api_key:
        console.print("[red]Error: No API key. Set in config.yaml or --api-key[/red]")
        raise typer.Exit(1)

    asyncio.run(_run_classify(source, _api_base, _api_key, _model, output, limit))


async def _run_classify(source: str, api_base: str, api_key: str, model: str, output: str | None, limit: int):
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
    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for i, article in enumerate(articles):
            task = progress.add_task(f"Classifying [{i+1}/{len(articles)}]: {article.title[:50]}...", total=None)
            try:
                result = await classify_article(article, api_base, api_key, model)
                results.append(result)
            except Exception as e:
                console.print(f"  [red]✗ Error: {e}[/red]")
            progress.update(task, completed=True)

    # Step 3: Display results
    _display_results(results)

    # Step 4: Save if requested
    if output:
        _save_results(results, output)

    # Summary
    relevant_count = sum(1 for r in results if r.relevant)
    avg_conf = sum(r.confidence for r in results) / len(results) if results else 0
    console.print(f"\n[bold]📊 Summary:[/bold] {relevant_count}/{len(results)} relevant | Avg confidence: {avg_conf:.2f}")


def _display_results(results):
    """Show results in a nice table."""
    table = Table(title="Classification Results", show_lines=True)
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


def _save_results(results, output_path):
    """Save results to JSON file."""
    summary = RunSummary(
        total=len(results),
        relevant=sum(1 for r in results if r.relevant),
        not_relevant=sum(1 for r in results if not r.relevant),
        avg_confidence=sum(r.confidence for r in results) / len(results) if results else 0,
        results=results,
    )
    with open(output_path, "w") as f:
        f.write(summary.model_dump_json(indent=2))
    console.print(f"\n[green]💾 Results saved to: {output_path}[/green]")


if __name__ == "__main__":
    app()
