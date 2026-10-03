# jev-imdb-benchmark

> Can a non-LLM model understand movie reviews? Let's find out.

[Jev](https://typesafe.ai) is TypeSafe AI's **System One** model. It doesn't generate text — it classifies, routes, and scores, returning typed answers with calibrated probabilities in a single pass. This project benchmarks it against 20 real IMDB reviews with known labels.

## Results

Jev analyzed 20 randomly sampled reviews from Stanford's [IMDB dataset](https://huggingface.co/datasets/stanfordnlp/imdb) (25K labeled reviews). Here's what happened:

| Metric | Value |
|---|---|
| Strict accuracy (exact match) | **85%** (17/20) |
| Lenient accuracy (mixed = ok) | **90%** (18/20) |
| Avg latency per call | **368ms** |
| Total input tokens | 13,474 |
| Total cost | **$0.000566** |

### Why two accuracy numbers?

IMDB labels are binary (positive/negative), but real reviews are often nuanced. Jev returned "mixed" for 1 review — one that contained both praise and criticism. The only true misclassifications were 2 reviews where Jev picked the opposite sentiment.

Full per-review breakdown with latency, cost, confidence, spoiler detection, and quality scores is saved to `results.csv` after each run.

## How it works

Each review is sent to Jev with three typed questions in a single `system_one()` call:

```python
questions = {
    "sentiment": Choice(...)    # positive / negative / mixed
    "is_spoiler": Noul(...)     # boolean probability
    "quality": Score(...)       # 1-3 scale
}
```

| Question type | What Jev returns |
|---|---|
| **Choice** | The best option + confidence + per-option probabilities |
| **Noul** | A probability (0.0 to 1.0) for a yes/no question |
| **Score** | A position on a scale + confidence + per-level probabilities |

## Quick start

```bash
git clone https://github.com/rahulbansalc6414/jev-imdb-benchmark.git
cd jev-imdb-benchmark

# Add your TypeSafe API key
cp .env.example .env
# Edit .env → TYPESAFE_API_KEY=your-key-here

# Run
uv run python main.py
```

Get an API key at [console.typesafe.ai](https://console.typesafe.ai).

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/)

## Project structure

```
main.py            # Entrypoint — loads IMDB data, runs the loop
jev_analyzer.py    # Jev client + analyze_review()
display.py         # Rich terminal output + CSV export
```

## Cost

Jev charges **$0.042 per million input tokens** with free output. A full 20-review run uses ~14K tokens and costs under **$0.001** — less than a tenth of a cent.
