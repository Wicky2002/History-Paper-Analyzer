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


def parse_sinhala_justification(text: str) -> dict[str, str]:
    """Parse Sinhala justification into structured sections."""
    sections = {
        "final_score": "",
        "breakdown": "",
        "ontology": "",
        "missing_concepts": "",
        "evidence_coverage": "",
        "confidence": "",
        "notes": "",
    }
    
    lines = text.split("\n")
    current_section = None
    section_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if "ලකුණු:" in line and "සංශෝධිත" not in line:
            if section_lines and current_section:
                sections[current_section] = "\n".join(section_lines)
            current_section = "final_score"
            section_lines = [line]
        elif "විස්තරය:" in line:
            if section_lines and current_section:
                sections[current_section] = "\n".join(section_lines)
            current_section = "breakdown"
            section_lines = [line]
        elif "ඔන්ටොලොජි" in line:
            if section_lines and current_section:
                sections[current_section] = "\n".join(section_lines)
            current_section = "ontology"
            section_lines = [line]
        elif "අහිමි" in line or "කරුණු:" in line:
            if section_lines and current_section:
                sections[current_section] = "\n".join(section_lines)
            current_section = "missing_concepts"
            section_lines = [line]
        elif "සාක්ෂි" in line and "ආවරණය" in line:
            if section_lines and current_section:
                sections[current_section] = "\n".join(section_lines)
            current_section = "evidence_coverage"
            section_lines = [line]
        elif "විශ්වාස" in line and "මට්ටම" in line:
            if section_lines and current_section:
                sections[current_section] = "\n".join(section_lines)
            current_section = "confidence"
            section_lines = [line]
        elif "සටහන" in line:
            if section_lines and current_section:
                sections[current_section] = "\n".join(section_lines)
            current_section = "notes"
            section_lines = [line]
        else:
            if current_section:
                section_lines.append(line)
    
    if section_lines and current_section:
        sections[current_section] = "\n".join(section_lines)
    
    return sections


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


def render_justification_sections(justification: str) -> None:
    sections = parse_sinhala_justification(justification)

    if sections["final_score"]:
        st.success(sections["final_score"])

    summary_cols = st.columns(3)
    with summary_cols[0]:
        st.markdown("### ලකුණු")
        st.write(sections["final_score"] or "-")
    with summary_cols[1]:
        st.markdown("### සාක්ෂි")
        st.write(sections["evidence_coverage"] or "-")
    with summary_cols[2]:
        st.markdown("### විශ්වාසය")
        st.write(sections["confidence"] or "-")

    if sections["breakdown"]:
        with st.container(border=True):
            st.markdown("#### 🧩 මාපක බිඳුම්")
            st.write(sections["breakdown"])

    if sections["ontology"]:
        with st.container(border=True):
            st.markdown("#### 🔗 ඔන්ටොලොජි තර්කය")
            st.write(sections["ontology"])

    if sections["missing_concepts"]:
        with st.container(border=True):
            st.markdown("#### ❌ අහිමි කරුණු")
            st.write(sections["missing_concepts"])

    if sections["notes"]:
        with st.container(border=True):
            st.markdown("#### 📌 සටහන")
            st.write(sections["notes"])


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

    if st.button("Evaluate", use_container_width=True):
        if not answer_text.strip():
            st.warning("කරුණාකර පිළිතුර ඇතුළත් කරන්න.")
            return

        marking_guide = json.dumps(selected_question, ensure_ascii=False)
        
        # Show progress with st.status
        with st.status("🔄 Evaluating answer...", expanded=True) as status:
            status_messages = []
            
            def update_status(step_name: str, message: str, emoji: str):
                status.write(f"{emoji} {step_name}: {message}")
            
            try:
                result = run_single_case(
                    question=selected_question["question_si"],
                    student_answer=answer_text.strip(),
                    marking_guide=marking_guide,
                    status_callback=update_status,
                )
                status.update(label="✅ Evaluation Complete!", state="complete")
            except Exception as e:
                status.update(label="❌ Error during evaluation", state="error")
                st.error(f"Evaluation failed: {str(e)}")
                return

        # Display results in sections
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("ලකුණු", f"{result.get('final_score', 0)}/20")
        with col2:
            st.metric("සංශෝධිතයි", f"{result.get('confidence_adjusted_score', 0)}/20")
        with col3:
            confidence = result.get('retrieval_confidence', 0.0)
            st.metric("විශ්වාස", f"{confidence:.3f}")
        with col4:
            coverage_text = result.get('evidence_coverage', '0/0')
            if "/" in coverage_text:
                grounded, total = coverage_text.split()[0].split("/")
                pct = (int(grounded) / int(total) * 100) if int(total) > 0 else 0
                st.metric("ඉතිරි", f"{pct:.0f}%")

        st.divider()

        # Breakdown Table
        st.subheader("📋 විස්තරිත බිඳුම්")
        breakdown_rows = build_breakdown_rows(result)
        if breakdown_rows:
            st.table(breakdown_rows)
        else:
            st.info("විස්තරණ තොරතුරු නොමැත.")

        # Ontology Section
        st.subheader("🔗 ඔන්ටොලොජි සම්බන්ධතා")
        ontology_rows = build_ontology_rows(result)
        if ontology_rows:
            st.table(ontology_rows)
        else:
            st.info("ඔන්ටොලොජි සම්බන්ධතා හමු නොවීය.")

        # Parse and display Sinhala explanation in compact sections
        st.subheader("📝 සිංහල පැහැදිලි කිරීම")
        justification = result.get("justification", "")

        if justification:
            render_justification_sections(justification)
        else:
            st.warning("පැහැදිලි කිරීම උත්පාදනය කිරීමට නොහැකි විය.")


if __name__ == "__main__":
    main()
