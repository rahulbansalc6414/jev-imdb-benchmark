import csv
import os

from display import _is_lenient_match, save_csv


def test_lenient_match_exact():
    assert _is_lenient_match("positive", "positive") is True
    assert _is_lenient_match("negative", "negative") is True


def test_lenient_match_mixed_is_ok():
    assert _is_lenient_match("mixed", "positive") is True
    assert _is_lenient_match("mixed", "negative") is True


def test_lenient_match_wrong_polarity():
    assert _is_lenient_match("positive", "negative") is False
    assert _is_lenient_match("negative", "positive") is False


def test_save_csv(tmp_path):
    results = [
        {
            "text": "Great movie, loved it!",
            "dataset_index": 42,
            "ground_truth": "positive",
            "sentiment": "positive",
            "sentiment_confidence": 0.95,
            "is_spoiler": 0.1,
            "quality_score": 2.0,
            "input_tokens": 50,
            "latency_ms": 300.0,
            "cost_usd": 0.0000021,
        },
        {
            "text": "Terrible film, waste of time.",
            "dataset_index": 99,
            "ground_truth": "negative",
            "sentiment": "mixed",
            "sentiment_confidence": 0.7,
            "is_spoiler": 0.8,
            "quality_score": 1.0,
            "input_tokens": 45,
            "latency_ms": 250.0,
            "cost_usd": 0.0000019,
        },
    ]

    csv_path = str(tmp_path / "test_results.csv")
    save_csv(results, csv_path)

    assert os.path.exists(csv_path)

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 2
    assert rows[0]["dataset_index"] == "42"
    assert rows[0]["jev_sentiment"] == "positive"
    assert rows[0]["strict_correct"] == "True"
    assert rows[0]["review_text"] == "Great movie, loved it!"
    assert rows[1]["jev_sentiment"] == "mixed"
    assert rows[1]["strict_correct"] == "False"
    assert rows[1]["lenient_correct"] == "True"
