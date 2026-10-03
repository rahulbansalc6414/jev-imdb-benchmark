import random
import time

from datasets import load_dataset
from dotenv import load_dotenv
from rich.console import Console

from display import print_detail_table, print_summary, save_csv
from jev_analyzer import analyze_review, create_client

load_dotenv()

SAMPLE_SIZE = 20
RANDOM_SEED = 42


def load_reviews() -> list[dict]:
    ds = load_dataset("stanfordnlp/imdb", split="test")
    positive = [r for r in ds if r["label"] == 1]
    negative = [r for r in ds if r["label"] == 0]
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(positive)
    rng.shuffle(negative)
    reviews = positive[:SAMPLE_SIZE // 2] + negative[:SAMPLE_SIZE // 2]
    rng.shuffle(reviews)
    for r in reviews:
        r["ground_truth"] = "positive" if r["label"] == 1 else "negative"
    return reviews


def main():
    console = Console()
    client = create_client()

    console.print("\n[bold]Loading IMDB reviews from Hugging Face...[/bold]")
    reviews = load_reviews()

    console.print(f"Analyzing {len(reviews)} reviews with Jev...\n")

    results = []
    start = time.perf_counter()

    for i, review in enumerate(reviews):
        analysis = analyze_review(client, review["text"])
        results.append({**review, **analysis})
        console.print(f"  [{i + 1}/{len(reviews)}] {review['ground_truth']:>8} → jev says {analysis['sentiment']:<8} (confidence: {analysis['sentiment_confidence']:.0%})")

    elapsed = time.perf_counter() - start

    print_summary(console, results, elapsed)
    print_detail_table(console, results)
    csv_path = save_csv(results)
    console.print(f"\n[bold]Results saved to {csv_path}[/bold]\n")


if __name__ == "__main__":
    main()
