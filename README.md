# jev-imdb-benchmark

> Can a non-LLM model understand movie reviews? Let's find out.

[Jev](https://typesafe.ai) is TypeSafe AI's **System One** model. It doesn't generate text — it classifies, routes, and scores, returning typed answers with calibrated probabilities in a single pass. This project benchmarks it against real IMDB reviews with known labels.

## Results

Jev analyzed 50 randomly sampled reviews from Stanford's [IMDB dataset](https://huggingface.co/datasets/stanfordnlp/imdb) (25K labeled reviews). Here's what happened:

| Metric | Value |
|---|---|
| Strict accuracy (exact match) | **84%** (42/50) |
| Lenient accuracy (mixed = ok) | **94%** (47/50) |
| Avg latency per call | **361ms** |
| Total input tokens | 32,782 |
| Total cost | **$0.001377** |

### Why two accuracy numbers?

IMDB labels are binary (positive/negative), but real reviews are often nuanced. Jev returned "mixed" for 5 reviews — ones that contained both praise and criticism. The actual wrong-polarity errors (positive ↔ negative) were only 3 out of 50.

Full per-review breakdown with latency, cost, confidence, spoiler detection, quality scores, and the complete review text is saved to `results.csv` after each run.

![Sample output](screenshot.png)

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

# Run (default: 20 reviews)
uv run python main.py

# Or specify a sample size
uv run python main.py -n 50
```

Get an API key at [console.typesafe.ai](https://console.typesafe.ai).

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/)

## Project structure

```
main.py            # Entrypoint — loads IMDB data, runs the loop
jev_analyzer.py    # Jev client + analyze_review()
display.py         # Rich terminal output + CSV export
test_display.py    # Tests for scoring logic and CSV output
```

## Cost

Jev charges **$0.042 per million input tokens** with free output. A 20-review run costs ~$0.0006, a 50-review run costs ~$0.0014 — both well under a penny.
