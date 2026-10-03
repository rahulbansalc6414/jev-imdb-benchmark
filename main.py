import argparse
import random
import time

from datasets import load_dataset
from dotenv import load_dotenv
from rich.console import Console

from display import print_detail_table, print_summary, save_csv
from jev_analyzer import analyze_review, create_client

load_dotenv()

DEFAULT_SAMPLE_SIZE = 20
RANDOM_SEED = 42


def load_reviews(sample_size: int) -> list[dict]:
    ds = load_dataset("stanfordnlp/imdb", split="test")
    indexed = [{"text": r["text"], "label": r["label"], "dataset_index": i} for i, r in enumerate(ds)]
    positive = [r for r in indexed if r["label"] == 1]
    negative = [r for r in indexed if r["label"] == 0]
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(positive)
    rng.shuffle(negative)
    half = sample_size // 2
    reviews = positive[:half] + negative[:half]
    rng.shuffle(reviews)
    for r in reviews:
        r["ground_truth"] = "positive" if r["label"] == 1 else "negative"
    return reviews


def main():
    parser = argparse.ArgumentParser(description="Benchmark Jev on IMDB reviews")
    parser.add_argument("-n", "--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
                        help=f"Number of reviews to analyze (default: {DEFAULT_SAMPLE_SIZE})")
    args = parser.parse_args()

    console = Console()
    client = create_client()

    console.print("\n[bold]Loading IMDB reviews from Hugging Face...[/bold]")
    reviews = load_reviews(args.sample_size)

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
