#!/usr/bin/env python3
"""Quick test of the fixes: confidence capping and status callbacks."""

import json
from agents.workflow import run_single_case

# Load sample question
with open("data/question_bank_si.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

q = questions[0]

# Test with status callback
print("Testing workflow with status callback...\n")

steps = []
def capture_status(step_name, message, emoji):
    steps.append(f"{emoji} {step_name}: {message}")
    print(f"{emoji} {step_name}: {message}")

result = run_single_case(
    question=q['question_si'],
    student_answer=q['sample_answers']['correct_100'],
    marking_guide=json.dumps(q, ensure_ascii=False),
    status_callback=capture_status,
)

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"✅ Final Score: {result['final_score']}/20")
print(f"✅ Adjusted Score: {result['confidence_adjusted_score']}/20")
print(f"✅ Confidence: {result['retrieval_confidence']:.4f} (should be ≤ 1.0)")
assert result['retrieval_confidence'] <= 1.0, f"ERROR: Confidence {result['retrieval_confidence']} exceeds 1.0!"
print(f"✅ Evidence Coverage: {result['evidence_coverage']}")
print(f"\n✅ All {len(steps)} workflow steps executed successfully with callbacks!")
