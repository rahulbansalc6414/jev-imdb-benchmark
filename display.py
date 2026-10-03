import csv

from rich.console import Console
from rich.table import Table


def _is_lenient_match(jev_sentiment: str, ground_truth: str) -> bool:
    if jev_sentiment == ground_truth:
        return True
    if jev_sentiment == "mixed":
        return True
    return False


def print_summary(console: Console, results: list[dict], elapsed: float):
    strict = sum(1 for r in results if r["sentiment"] == r["ground_truth"])
    lenient = sum(1 for r in results if _is_lenient_match(r["sentiment"], r["ground_truth"]))
    total_cost = sum(r["cost_usd"] for r in results)
    total_tokens = sum(r["input_tokens"] for r in results)
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)
    mixed_count = sum(1 for r in results if r["sentiment"] == "mixed")

    table = Table(title="\nResults Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Reviews analyzed", str(len(results)))
    table.add_row("Strict accuracy", f"{strict / len(results):.0%} ({strict}/{len(results)})")
    table.add_row("Lenient accuracy", f"{lenient / len(results):.0%} ({lenient}/{len(results)}) [dim](mixed = ok)[/dim]")
    table.add_row("Mixed predictions", str(mixed_count))
    table.add_row("Total time", f"{elapsed:.1f}s")
    table.add_row("Avg latency per call", f"{avg_latency:.0f}ms")
    table.add_row("Total input tokens", f"{total_tokens:,}")
    table.add_row("Total cost", f"${total_cost:.6f}")
    console.print(table)

    spoiler_count = sum(1 for r in results if r["is_spoiler"] > 0.5)
    avg_quality = sum(r["quality_score"] for r in results) / len(results)
    console.print(f"\n  Spoiler reviews detected: {spoiler_count}/{len(results)}")
    console.print(f"  Average review quality:   {avg_quality:.2f}/3.00\n")


def print_detail_table(console: Console, results: list[dict]):
    detail = Table(title="Per-Review Breakdown")
    detail.add_column("#", justify="right")
    detail.add_column("IMDB ID", justify="right")
    detail.add_column("Ground Truth")
    detail.add_column("Jev Sentiment")
    detail.add_column("Confidence", justify="right")
    detail.add_column("Spoiler?", justify="center")
    detail.add_column("Quality", justify="right")
    detail.add_column("Latency", justify="right")
    detail.add_column("Cost", justify="right")
    detail.add_column("Preview")

    for i, r in enumerate(results):
        match_icon = "[green]✓[/green]" if r["sentiment"] == r["ground_truth"] else "[red]✗[/red]"
        spoiler = "[yellow]yes[/yellow]" if r["is_spoiler"] > 0.5 else "no"
        preview = r["text"][:40].replace("\n", " ") + "..."
        detail.add_row(
            f"{i + 1}",
            str(r["dataset_index"]),
            r["ground_truth"],
            f"{match_icon} {r['sentiment']}",
            f"{r['sentiment_confidence']:.0%}",
            spoiler,
            f"{r['quality_score']:.1f}",
            f"{r['latency_ms']:.0f}ms",
            f"${r['cost_usd']:.7f}",
            preview,
        )

    console.print(detail)


def save_csv(results: list[dict], path: str = "results.csv"):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "review_num", "dataset_index", "ground_truth", "jev_sentiment",
            "sentiment_confidence", "is_spoiler_prob", "quality_score",
            "input_tokens", "latency_ms", "cost_usd", "strict_correct",
            "lenient_correct", "review_text",
        ])
        writer.writeheader()
        for i, r in enumerate(results):
            writer.writerow({
                "review_num": i + 1,
                "dataset_index": r["dataset_index"],
                "ground_truth": r["ground_truth"],
                "jev_sentiment": r["sentiment"],
                "sentiment_confidence": f"{r['sentiment_confidence']:.4f}",
                "is_spoiler_prob": f"{r['is_spoiler']:.4f}",
                "quality_score": f"{r['quality_score']:.2f}",
                "input_tokens": r["input_tokens"],
                "latency_ms": f"{r['latency_ms']:.1f}",
                "cost_usd": f"{r['cost_usd']:.8f}",
                "strict_correct": r["sentiment"] == r["ground_truth"],
                "lenient_correct": _is_lenient_match(r["sentiment"], r["ground_truth"]),
                "review_text": r["text"],
            })
    return path
