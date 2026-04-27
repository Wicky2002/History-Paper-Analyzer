import json
import os
from typing import Any, Mapping

import streamlit as st

from agents.workflow import run_single_case

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUESTION_BANK_PATH = os.path.join(BASE_DIR, "data", "question_bank_si.json")


def load_questions() -> list[dict[str, Any]]:
    with open(QUESTION_BANK_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError("ප්‍රශ්න ගොනුව වැරදියි.")
    return payload


def build_breakdown_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in result.get("scoring_output", {}).get("breakdown", []):
        rows.append(
            {
                "මාපකය": item.get("criterion", ""),
                "ලැබුණු ලකුණු": f"{item.get('awarded', 0)}/{item.get('max_marks', 0)}",
                "තත්ත්වය": item.get("reason", ""),
                "සාක්ෂි chunk": ", ".join(item.get("evidence_chunk_ids", [])) or "නොමැත",
            }
        )
    return rows


def build_ontology_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for rel in result.get("ontology_match_output", []):
        rows.append(
            {
                "Entity": rel.get("subject", ""),
                "Relation": rel.get("relation", ""),
                "Expected": rel.get("expected", ""),
                "Student": rel.get("student", ""),
                "Classification": rel.get("label", ""),
                "Chunk": rel.get("chunk_id", "නොමැත"),
            }
        )
    return rows


def main() -> None:
    st.set_page_config(page_title="සිංහල ඉතිහාස උත්තර ඇගයීම", layout="wide")
    st.title("Offline Sinhala Open-Ended Answer Scorer")
    st.markdown("අනුරාධපුර යුගය | පාඨපොත්-පදනම් RAG + ඔන්ටොලොජි + නියමිත ලකුණු නීති")

    questions = load_questions()
    options = {
        f"Q{q['id']} - {q['question_si']}": q
        for q in questions
    }

    selected_label = st.selectbox("ප්‍රශ්නය තෝරන්න", list(options.keys()))
    selected_question = options[selected_label]

    st.markdown("### ප්‍රශ්නය")
    st.markdown(f"**{selected_question['question_si']}**")

    parts = selected_question.get("parts", {})
    st.markdown("### කොටස්")
    st.markdown(f"**(අ)** {parts.get('a', '')}")
    st.markdown(f"**(ආ)** {parts.get('b', '')}")
    st.markdown(f"**(ඇ)** {parts.get('c', '')}")

    with st.expander("ඩෙමෝ සඳහා නියැදි පිළිතුරු 3"):
        samples = selected_question.get("sample_answers", {})
        st.markdown("**100% නිවැරදි පිළිතුර:**")
        st.markdown(samples.get("correct_100", samples.get("high", "-")))
        st.markdown("**මිශ්‍ර (අර්ධ නිවැරදි) පිළිතුර:**")
        st.markdown(samples.get("mixed", samples.get("medium", "-")))
        st.markdown("**100% වැරදි පිළිතුර:**")
        st.markdown(samples.get("wrong_100", samples.get("low", "-")))

    answer_text = st.text_area("ශිෂ්‍ය පිළිතුර (සිංහල)", height=180)

    if st.button("Evaluate"):
        if not answer_text.strip():
            st.warning("කරුණාකර පිළිතුර ඇතුළත් කරන්න.")
            return

        marking_guide = json.dumps(selected_question, ensure_ascii=False)
        result = run_single_case(
            question=selected_question["question_si"],
            student_answer=answer_text.strip(),
            marking_guide=marking_guide,
        )

        st.markdown("## ප්‍රතිඵල")
        st.markdown(f"**Final Score:** {result.get('final_score', 0)}/20")
        st.markdown(f"**Adjusted Score:** {result.get('confidence_adjusted_score', 0)}/20")
        st.markdown(f"**Retrieval Confidence:** {result.get('retrieval_confidence', 0.0):.4f}")
        st.markdown(f"**සාක්ෂි ආවරණය:** {result.get('evidence_coverage', '0/0 වාක්‍ය පාඨපොත් සාක්ෂි සමඟ සම්බන්ධයි')}")

        st.markdown("### Breakdown Table")
        st.table(build_breakdown_rows(result))

        st.markdown("### Ontology Section")
        ontology_rows = build_ontology_rows(result)
        if ontology_rows:
            st.table(ontology_rows)
        else:
            st.markdown("ඔන්ටොලොජි සම්බන්ධතා හමු නොවීය.")

        st.markdown("### Explanation (සිංහල)")
        st.markdown(result.get("justification", ""))


if __name__ == "__main__":
    main()
