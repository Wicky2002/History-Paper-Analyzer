import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.workflow import run_single_case

QUESTION_BANK = ROOT / "data" / "question_bank_si.json"
OUT_PATH = ROOT / "data" / "processed" / "evaluation_results_demo.json"


def pearson_corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    den = den_x * den_y
    if den == 0:
        return 0.0
    return num / den


def parse_coverage(text: str) -> tuple[int, int]:
    # Format: X/Y වාක්‍ය ...
    if not text or "/" not in text:
        return 0, 0
    left = text.split(" ", 1)[0]
    if "/" not in left:
        return 0, 0
    a, b = left.split("/", 1)
    try:
        return int(a), int(b)
    except ValueError:
        return 0, 0


def main() -> None:
    qbank = json.loads(QUESTION_BANK.read_text(encoding="utf-8"))

    key_expected = {
        "correct_100": 20.0,
        "mixed": 10.0,
        "wrong_100": 0.0,
    }

    cases = []
    y_true = []
    y_raw = []
    y_adj = []
    latencies = []

    grounded_sum = 0
    total_sum = 0

    ont_full = 0
    ont_partial = 0
    ont_wrong = 0

    missing_alignment_hits = 0
    missing_alignment_total = 0

    for q in qbank:
        for key, expected in key_expected.items():
            ans = q.get("sample_answers", {}).get(key)
            if not ans:
                continue

            start = time.perf_counter()
            result = run_single_case(
                question=q["question_si"],
                student_answer=ans,
                marking_guide=json.dumps(q, ensure_ascii=False),
            )
            elapsed = time.perf_counter() - start

            raw = float(result.get("final_score", 0.0))
            adj = float(result.get("confidence_adjusted_score", 0.0))
            conf = float(result.get("retrieval_confidence", 0.0))

            cov_g, cov_t = parse_coverage(result.get("evidence_coverage", ""))
            grounded_sum += cov_g
            total_sum += cov_t

            labels = [
                rel.get("label", "")
                for rel in result.get("ontology_match_output", [])
                if isinstance(rel, dict)
            ]
            for lbl in labels:
                if "✔" in lbl:
                    ont_full += 1
                elif "~" in lbl:
                    ont_partial += 1
                elif "✘" in lbl:
                    ont_wrong += 1

            missing = result.get("missing_concepts", []) or []
            expected_missing = key in {"mixed", "wrong_100"}
            if expected_missing:
                missing_alignment_hits += 1 if len(missing) > 0 else 0
            else:
                missing_alignment_hits += 1 if len(missing) == 0 else 0
            missing_alignment_total += 1

            latencies.append(elapsed)
            y_true.append(expected)
            y_raw.append(raw)
            y_adj.append(adj)

            cases.append(
                {
                    "question_id": q.get("id"),
                    "answer_type": key,
                    "expected": expected,
                    "raw_score": raw,
                    "adjusted_score": adj,
                    "retrieval_confidence": conf,
                    "evidence_coverage": result.get("evidence_coverage", ""),
                    "missing_count": len(missing),
                    "latency_sec": round(elapsed, 3),
                }
            )
            print(
                f"Q{q.get('id')} {key}: raw={raw:.1f}, adj={adj:.1f}, conf={conf:.4f}, latency={elapsed:.2f}s"
            )

    n = len(y_true)
    mae_raw = sum(abs(a - b) for a, b in zip(y_raw, y_true)) / n if n else 0.0
    rmse_raw = math.sqrt(sum((a - b) ** 2 for a, b in zip(y_raw, y_true)) / n) if n else 0.0
    corr_raw = pearson_corr(y_raw, y_true)

    mae_adj = sum(abs(a - b) for a, b in zip(y_adj, y_true)) / n if n else 0.0
    rmse_adj = math.sqrt(sum((a - b) ** 2 for a, b in zip(y_adj, y_true)) / n) if n else 0.0
    corr_adj = pearson_corr(y_adj, y_true)

    exact_raw = sum(1 for a, b in zip(y_raw, y_true) if abs(a - b) < 1e-9)
    exact_adj = sum(1 for a, b in zip(y_adj, y_true) if abs(a - b) < 1e-9)

    within2_raw = sum(1 for a, b in zip(y_raw, y_true) if abs(a - b) <= 2)
    within2_adj = sum(1 for a, b in zip(y_adj, y_true) if abs(a - b) <= 2)

    over_raw = sum(1 for a, b in zip(y_raw, y_true) if a > b)
    over_adj = sum(1 for a, b in zip(y_adj, y_true) if a > b)
    under_raw = sum(1 for a, b in zip(y_raw, y_true) if a < b)
    under_adj = sum(1 for a, b in zip(y_adj, y_true) if a < b)

    by_type = {}
    for t in key_expected:
        idxs = [i for i, c in enumerate(cases) if c["answer_type"] == t]
        if not idxs:
            continue
        ty = [y_true[i] for i in idxs]
        tr = [y_raw[i] for i in idxs]
        by_type[t] = {
            "count": len(idxs),
            "expected_mean": round(sum(ty) / len(ty), 3),
            "pred_mean": round(sum(tr) / len(tr), 3),
            "mae": round(sum(abs(a - b) for a, b in zip(tr, ty)) / len(idxs), 3),
        }

    summary = {
        "n_cases": n,
        "metrics_raw": {
            "mae": round(mae_raw, 4),
            "rmse": round(rmse_raw, 4),
            "pearson": round(corr_raw, 4),
            "exact_match_rate": round((exact_raw / n) * 100, 2) if n else 0.0,
            "within_2_rate": round((within2_raw / n) * 100, 2) if n else 0.0,
            "over_scoring_cases": over_raw,
            "under_scoring_cases": under_raw,
        },
        "metrics_adjusted": {
            "mae": round(mae_adj, 4),
            "rmse": round(rmse_adj, 4),
            "pearson": round(corr_adj, 4),
            "exact_match_rate": round((exact_adj / n) * 100, 2) if n else 0.0,
            "within_2_rate": round((within2_adj / n) * 100, 2) if n else 0.0,
            "over_scoring_cases": over_adj,
            "under_scoring_cases": under_adj,
        },
        "by_answer_type": by_type,
        "evidence_coverage": {
            "grounded": grounded_sum,
            "total": total_sum,
            "ratio_percent": round((grounded_sum / total_sum) * 100, 2) if total_sum else 0.0,
        },
        "ontology_labels": {
            "full": ont_full,
            "partial": ont_partial,
            "wrong": ont_wrong,
            "full_percent": round((ont_full / max(1, (ont_full + ont_partial + ont_wrong))) * 100, 2),
            "partial_percent": round((ont_partial / max(1, (ont_full + ont_partial + ont_wrong))) * 100, 2),
            "wrong_percent": round((ont_wrong / max(1, (ont_full + ont_partial + ont_wrong))) * 100, 2),
        },
        "missing_concept_alignment_percent": round((missing_alignment_hits / max(1, missing_alignment_total)) * 100, 2),
        "latency_sec": {
            "mean": round(sum(latencies) / max(1, len(latencies)), 3),
            "median": round(statistics.median(latencies), 3) if latencies else 0.0,
            "min": round(min(latencies), 3) if latencies else 0.0,
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
    }

    out = {"summary": summary, "cases": cases}
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
