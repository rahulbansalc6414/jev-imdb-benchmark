import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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

    results = [None] * len(reviews)
    completed = 0
    start = time.perf_counter()

    def process(idx, review):
        analysis = analyze_review(client, review["text"])
        return idx, {**review, **analysis}

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(process, i, r): i for i, r in enumerate(reviews)}
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result
            completed += 1
            console.print(f"  [{completed}/{len(reviews)}] {result['ground_truth']:>8} → jev says {result['sentiment']:<8} (confidence: {result['sentiment_confidence']:.0%})")

    elapsed = time.perf_counter() - start

    print_summary(console, results, elapsed)
    print_detail_table(console, results)
    csv_path = save_csv(results, path=f"results/results_{args.sample_size}.csv")
    console.print(f"\n[bold]Results saved to {csv_path}[/bold]\n")


if __name__ == "__main__":
    main()
