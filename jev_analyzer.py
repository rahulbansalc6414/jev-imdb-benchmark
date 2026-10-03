import time

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

COST_PER_M_INPUT_TOKENS = 0.042


def create_client() -> TypeSafeClient:
    return TypeSafeClient()


def analyze_review(client: TypeSafeClient, text: str) -> dict:
    t0 = time.perf_counter()
    response = client.system_one(
        state=text[:2000],
        questions={
            "sentiment": Choice(
                instructions="What is the overall sentiment of this movie review?",
                criteria={
                    "positive": "The reviewer liked or recommended the movie",
                    "negative": "The reviewer disliked or criticized the movie",
                    "mixed": "The reviewer had both positive and negative feelings",
                },
            ),
            "is_spoiler": Noul(
                instructions="Does this review reveal major plot points or endings?",
            ),
            "quality": Score(
                instructions="How well-written and substantive is this review?",
                criteria=[
                    "Low effort, vague, or uninformative",
                    "Decent but surface-level",
                    "Thoughtful with specific examples or reasoning",
                ],
            ),
        },
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    input_tokens = response.usage.input_tokens
    cost = input_tokens * COST_PER_M_INPUT_TOKENS / 1_000_000
    return {
        "sentiment": response.answers["sentiment"].choice,
        "sentiment_confidence": response.answers["sentiment"].confidence,
        "is_spoiler": response.answers["is_spoiler"].noul,
        "quality_score": response.answers["quality"].score,
        "input_tokens": input_tokens,
        "latency_ms": latency_ms,
        "cost_usd": cost,
    }
