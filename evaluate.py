#!/usr/bin/env python3
"""
MM-OCEAN Benchmark Evaluation Script

Self-contained scorer: given a model's raw outputs (JSON per video),
computes all metrics reported in the paper:
  - Task 1: Ordinal Personality Rating (Accuracy, MAE)
  - Task 2: Open-Ended Rating Reasoning (T2-Avg4 via AI-as-Judge)
  - Task 3: Structured Cue Grounding (overall + per-category MCQ accuracy)
  - Failure-mode rates: PR, CR, IR, HR
  - Rating-Grounding Misalignment (RGM)

Usage:
  python evaluate.py --gt_dir data/test --pred_file results/my_model.jsonl

The prediction file should be a JSONL with one JSON object per video,
matching the output schema of prompts/unified.py:
  {
    "video_id": "abc.mp4",
    "task1": {"extraversion": "High", ...},
    "task2": {"extraversion": "The person ...", ...},
    "task3": [{"q_idx": 1, "answer": "C"}, ...]
  }

If Task 2 has not been judged yet, run the judge first:
  python evaluate.py --gt_dir data/test --pred_file results/my_model.jsonl --judge --judge_model gpt-4o-mini
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

LEVELS = ["Very Low", "Low", "Medium", "High", "Very High"]
LEVEL_MAP = {l: i for i, l in enumerate(LEVELS)}
TRAITS = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]
MCQ_CATEGORIES = [
    "Personality Attribution", "Counterfactual Reasoning",
    "Temporal-Causal Reasoning", "Mixed Emotion Discrimination",
    "Micro-expression Localization", "Spatial Localization Verification",
    "Temporal-Spatial Joint Localization",
]

# Default binarization thresholds (Section 4.4 of the paper)
THETA_1 = 0.5   # T1: majority of 5 traits correct
THETA_2 = 0.7   # T2: judge composite >= 7/10
THETA_3 = 0.5   # T3: majority of MCQs correct


def load_gt(gt_dir: Path) -> dict:
    """Load ground-truth data from MM-OCEAN annotation JSONs."""
    gt = {}
    for f in sorted(gt_dir.glob("*.json")):
        d = json.load(open(f))
        vid = d["video_id"]
        # Discretize continuous scores to ordinal levels
        scores = d.get("original_scores", {})
        levels = {}
        for trait, score in scores.items():
            if trait not in TRAITS:
                continue
            if score <= 0.2:   levels[trait] = "Very Low"
            elif score <= 0.4: levels[trait] = "Low"
            elif score <= 0.6: levels[trait] = "Medium"
            elif score <= 0.8: levels[trait] = "High"
            else:              levels[trait] = "Very High"
        gt[vid] = {
            "levels": levels,
            "mcqs": d.get("mcq_questions", []),
            "observations": d.get("observations", []),
            "personality_analyses": d.get("personality_analyses", []),
        }
    return gt


def load_predictions(pred_file: Path) -> dict:
    """Load model predictions from a JSONL file."""
    preds = {}
    with open(pred_file) as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                preds[r["video_id"]] = r
            except Exception:
                continue
    return preds


# ── Task 1: Ordinal Rating ─────────────────────────────────────────────

def score_task1(gt: dict, preds: dict) -> dict:
    correct = 0
    total = 0
    abs_errors = []
    for vid in gt:
        if vid not in preds:
            continue
        gt_levels = gt[vid]["levels"]
        pred_t1 = preds[vid].get("task1", {})
        for trait in TRAITS:
            g = gt_levels.get(trait)
            p = pred_t1.get(trait)
            if g and p and g in LEVELS and p in LEVELS:
                total += 1
                if g == p:
                    correct += 1
                abs_errors.append(abs(LEVEL_MAP[g] - LEVEL_MAP[p]))
    acc = correct / total * 100 if total else 0
    mae = sum(abs_errors) / len(abs_errors) if abs_errors else 0
    return {"accuracy": round(acc, 1), "mae": round(mae, 2), "n_traits": total}


# ── Task 2: Reasoning Quality (requires judged scores) ─────────────────

def score_task2(preds: dict) -> dict:
    """Score Task 2 from pre-judged predictions.

    Each prediction should have a 'task2_judged' field:
      {"extraversion": {"evidence_coverage": 7, "logical_coherence": 8,
                         "directional_accuracy": 6, "grounding_accuracy": 7}, ...}
    """
    dims = ["evidence_coverage", "logical_coherence",
            "directional_accuracy", "grounding_accuracy"]
    scores = []
    for vid, pred in preds.items():
        judged = pred.get("task2_judged", {})
        for trait in TRAITS:
            s = judged.get(trait, {})
            vals = [s.get(d) for d in dims]
            if all(isinstance(v, (int, float)) for v in vals):
                scores.append(sum(vals) / len(vals))
    avg = sum(scores) / len(scores) if scores else 0
    return {"t2_avg4": round(avg, 2), "n_samples": len(scores)}


# ── Task 3: MCQ Accuracy ───────────────────────────────────────────────

def score_task3(gt: dict, preds: dict) -> dict:
    correct = 0
    total = 0
    per_cat = defaultdict(lambda: [0, 0])  # [correct, total]
    for vid in gt:
        if vid not in preds:
            continue
        gt_mcqs = gt[vid]["mcqs"]
        pred_t3 = preds[vid].get("task3", [])
        pred_map = {r.get("q_idx", i + 1): r.get("answer", "") for i, r in enumerate(pred_t3)}
        for i, q in enumerate(gt_mcqs):
            q_idx = i + 1
            gt_ans = q.get("correct_answer", "")
            pred_ans = pred_map.get(q_idx, "")
            cat = q.get("category", "Unknown")
            total += 1
            per_cat[cat][1] += 1
            if pred_ans.strip().upper() == gt_ans.strip().upper():
                correct += 1
                per_cat[cat][0] += 1
    acc = correct / total * 100 if total else 0
    cat_accs = {c: round(v[0] / max(v[1], 1) * 100, 1) for c, v in per_cat.items()}
    return {"accuracy": round(acc, 1), "n_mcqs": total, "per_category": cat_accs}


# ── Failure-Mode Rates (PR / CR / IR / HR) ─────────────────────────────

def compute_failure_rates(gt: dict, preds: dict,
                          t2_scores: dict | None = None,
                          theta1=THETA_1, theta2=THETA_2, theta3=THETA_3) -> dict:
    """Compute per-sample binarized outcomes and aggregate failure-mode rates.

    Returns PR, CR, IR, HR as defined in Section 4.4.
    """
    r1_pass = 0
    r1_pass_r3_fail = 0
    r1_pass_r2_fail = 0
    r3_pass = 0
    r3_pass_r1_fail = 0
    all_pass = 0
    n = 0

    dims = ["evidence_coverage", "logical_coherence",
            "directional_accuracy", "grounding_accuracy"]

    for vid in gt:
        if vid not in preds:
            continue
        gt_levels = gt[vid]["levels"]
        pred_t1 = preds[vid].get("task1", {})
        gt_mcqs = gt[vid]["mcqs"]
        pred_t3 = preds[vid].get("task3", [])

        # R1: fraction of 5 traits correct
        t1_correct = sum(1 for t in TRAITS if pred_t1.get(t) == gt_levels.get(t))
        R1 = t1_correct / len(TRAITS)

        # R2: T2-Avg4 / 10 (from judged scores)
        R2 = 0
        judged = preds[vid].get("task2_judged", {})
        t2_vals = []
        for trait in TRAITS:
            s = judged.get(trait, {})
            vals = [s.get(d) for d in dims]
            if all(isinstance(v, (int, float)) for v in vals):
                t2_vals.append(sum(vals) / len(vals))
        if t2_vals:
            R2 = (sum(t2_vals) / len(t2_vals)) / 10.0

        # R3: fraction of MCQs correct
        pred_map = {r.get("q_idx", i + 1): r.get("answer", "") for i, r in enumerate(pred_t3)}
        mcq_correct = sum(1 for i, q in enumerate(gt_mcqs)
                         if pred_map.get(i + 1, "").strip().upper() == q.get("correct_answer", "").strip().upper())
        R3 = mcq_correct / max(len(gt_mcqs), 1)

        # Binarize
        b1 = int(R1 >= theta1)
        b2 = int(R2 >= theta2)
        b3 = int(R3 >= theta3)

        n += 1
        if b1:
            r1_pass += 1
            if not b3:
                r1_pass_r3_fail += 1
            if not b2:
                r1_pass_r2_fail += 1
        if b3:
            r3_pass += 1
            if not b1:
                r3_pass_r1_fail += 1
        if b1 and b2 and b3:
            all_pass += 1

    pr = (r1_pass_r3_fail / r1_pass * 100) if r1_pass else 0
    cr = (r1_pass_r2_fail / r1_pass * 100) if r1_pass else 0
    ir = (r3_pass_r1_fail / r3_pass * 100) if r3_pass else 0
    hr = (all_pass / n * 100) if n else 0

    return {
        "PR": round(pr, 1),
        "CR": round(cr, 1),
        "IR": round(ir, 1),
        "HR": round(hr, 1),
        "n_videos": n,
    }


# ── Main ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="MM-OCEAN Benchmark Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--gt_dir", required=True, help="Path to data/test/ with GT annotation JSONs")
    p.add_argument("--pred_file", required=True, help="Path to model prediction JSONL")
    p.add_argument("--theta1", type=float, default=THETA_1)
    p.add_argument("--theta2", type=float, default=THETA_2)
    p.add_argument("--theta3", type=float, default=THETA_3)
    args = p.parse_args()

    gt = load_gt(Path(args.gt_dir))
    preds = load_predictions(Path(args.pred_file))
    print(f"Loaded {len(gt)} GT videos, {len(preds)} predictions")
    overlap = set(gt) & set(preds)
    print(f"Overlap: {len(overlap)} videos\n")

    if not overlap:
        print("No overlapping video IDs. Check your prediction file.")
        sys.exit(1)

    # Task 1
    t1 = score_task1(gt, preds)
    print(f"=== Task 1: Ordinal Personality Rating ===")
    print(f"  Accuracy: {t1['accuracy']}%  MAE: {t1['mae']}  (n={t1['n_traits']} trait predictions)\n")

    # Task 2
    t2 = score_task2(preds)
    if t2["n_samples"] > 0:
        print(f"=== Task 2: Open-Ended Rating Reasoning ===")
        print(f"  T2-Avg4: {t2['t2_avg4']}  (n={t2['n_samples']} judged samples)\n")
    else:
        print(f"=== Task 2: No judged scores found (add 'task2_judged' field) ===\n")

    # Task 3
    t3 = score_task3(gt, preds)
    print(f"=== Task 3: Structured Cue Grounding ===")
    print(f"  Overall accuracy: {t3['accuracy']}%  (n={t3['n_mcqs']} MCQs)")
    if t3["per_category"]:
        print(f"  Per-category:")
        for cat, acc in sorted(t3["per_category"].items()):
            print(f"    {cat:40s} {acc:5.1f}%")
    print()

    # Failure-mode rates
    fm = compute_failure_rates(gt, preds, theta1=args.theta1, theta2=args.theta2, theta3=args.theta3)
    print(f"=== Failure-Mode Rates (theta1={args.theta1}, theta2={args.theta2}, theta3={args.theta3}) ===")
    print(f"  HR (Holistic-Grounding):    {fm['HR']}%")
    print(f"  PR (Prejudice):             {fm['PR']}%")
    print(f"  CR (Confabulation):         {fm['CR']}%")
    print(f"  IR (Integration-failure):   {fm['IR']}%")
    print(f"  (n={fm['n_videos']} videos)\n")

    # Summary JSON
    summary = {
        "task1_accuracy": t1["accuracy"],
        "task1_mae": t1["mae"],
        "task2_avg4": t2["t2_avg4"],
        "task3_accuracy": t3["accuracy"],
        "task3_per_category": t3["per_category"],
        "HR": fm["HR"],
        "PR": fm["PR"],
        "CR": fm["CR"],
        "IR": fm["IR"],
    }
    print(f"=== Summary JSON ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
