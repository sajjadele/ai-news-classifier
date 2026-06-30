import asyncio
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cli import load_config
from classifier.classifier import classify_article
from classifier.models import Article


TEST_SET_PATH = ROOT / "eval" / "test_set.csv"
RESULTS_PATH = ROOT / "eval" / "results.md"


def safe_divide(a: float, b: float) -> float:
    return a / b if b else 0.0


def calculate_metrics(tp: int, tn: int, fp: int, fn: int) -> dict:
    accuracy = safe_divide(tp + tn, tp + tn + fp + fn)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * ((precision * recall) / (precision + recall))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


async def evaluate() -> None:
    config = load_config()

    api_base = config.get("api_base")
    api_key = config.get("api_key")
    model = config.get("model")
    proxy = config.get("proxy")

    rows = []

    with open(TEST_SET_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("No rows found in eval/test_set.csv")
        return

    predictions = []

    for idx, row in enumerate(rows):
        article = Article(
            title=row["title"],
            content=row["content"],
            source="eval",
        )

        result = await classify_article(
            article=article,
            article_id=idx,
            api_base=api_base,
            api_key=api_key,
            model=model,
            proxy=proxy,
        )

        expected = int(row["my_label"])
        predicted = 1 if result.relevant else 0

        predictions.append({
            "title": row["title"],
            "expected": expected,
            "predicted": predicted,
            "confidence": result.confidence,
            "reason": result.reason,
            "error": result.error,
        })

    tp = sum(1 for p in predictions if p["expected"] == 1 and p["predicted"] == 1)
    tn = sum(1 for p in predictions if p["expected"] == 0 and p["predicted"] == 0)
    fp = sum(1 for p in predictions if p["expected"] == 0 and p["predicted"] == 1)
    fn = sum(1 for p in predictions if p["expected"] == 1 and p["predicted"] == 0)

    metrics = calculate_metrics(tp, tn, fp, fn)

    false_positives = [p for p in predictions if p["expected"] == 0 and p["predicted"] == 1]
    false_negatives = [p for p in predictions if p["expected"] == 1 and p["predicted"] == 0]

    print(f"Accuracy : {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall   : {metrics['recall']:.3f}")
    print(f"F1 Score : {metrics['f1']:.3f}")

    print("\nFalse Positives:")
    if false_positives:
        for item in false_positives:
            print(f"- {item['title']}")
            print(f"  Reason: {item['reason']}")
    else:
        print("- None")

    print("\nFalse Negatives:")
    if false_negatives:
        for item in false_negatives:
            print(f"- {item['title']}")
            print(f"  Reason: {item['reason']}")
    else:
        print("- None")

    report_lines = [
        "# Evaluation Results",
        "",
        f"- Total samples: {len(predictions)}",
        f"- True positives: {tp}",
        f"- True negatives: {tn}",
        f"- False positives: {fp}",
        f"- False negatives: {fn}",
        "",
        "## Metrics",
        "",
        f"- Accuracy: {metrics['accuracy']:.3f}",
        f"- Precision: {metrics['precision']:.3f}",
        f"- Recall: {metrics['recall']:.3f}",
        f"- F1: {metrics['f1']:.3f}",
        "",
        "## False Positives",
        "",
    ]

    if false_positives:
        for item in false_positives:
            report_lines.extend([
                f"- {item['title']}",
                f"  - Reason: {item['reason']}",
                f"  - Confidence: {item['confidence']:.2f}",
            ])
    else:
        report_lines.append("- None")

    report_lines.extend([
        "",
        "## False Negatives",
        "",
    ])

    if false_negatives:
        for item in false_negatives:
            report_lines.extend([
                f"- {item['title']}",
                f"  - Reason: {item['reason']}",
                f"  - Confidence: {item['confidence']:.2f}",
            ])
    else:
        report_lines.append("- None")

    RESULTS_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\nSaved report to {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(evaluate())
