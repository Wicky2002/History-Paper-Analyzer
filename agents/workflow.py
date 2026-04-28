import json
import os
import re
from typing import Any, TypedDict, cast

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, StateGraph

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, "data", "vector_db")
ONTOLOGY_PATH = os.path.join(BASE_DIR, "data", "ontology", "history_ontology_si.json")


class GradeState(TypedDict):
    question: str
    student_answer: str
    marking_guide: str
    retrieved_chunks: list[dict[str, Any]]
    retrieval_confidence: float
    low_confidence: bool
    selected_rubric: dict[str, Any]
    entity_matches: list[dict[str, Any]]
    relationship_checks: list[dict[str, Any]]
    coverage_output: dict[str, Any]
    scoring_output: dict[str, Any]
    final_score: int
    confidence_adjusted_score: float
    missing_concepts: list[str]
    ontology_match_output: list[dict[str, Any]]
    evidence_coverage: str
    justification: str


def _load_ontology() -> dict[str, Any]:
    with open(ONTOLOGY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


ONTOLOGY = _load_ontology()
POLICY = ONTOLOGY.get("scoring_policies", {})

print("Loading offline embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
vector_db = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings,
    collection_name="history_processed_si",
)
llama = ChatOllama(model="llama3", temperature=0.0)


def _safe_json_extract(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
    return {}


def _extract_custom_rubric(marking_guide: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(marking_guide)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    marking_scheme = payload.get("marking_scheme")
    if not isinstance(marking_scheme, list) or not marking_scheme:
        return None

    criteria = []
    for idx, item in enumerate(marking_scheme, start=1):
        if not isinstance(item, dict):
            continue
        criteria.append(
            {
                "criterion_id": item.get("criterion_id", f"m{idx}"),
                "description": item.get("criterion", ""),
                "marks": int(item.get("marks", 0)),
                "concept_ids": item.get("concept_ids", []),
                "expected_terms": item.get("expected_terms", []),
                "part": item.get("part", ""),
            }
        )

    if not criteria:
        return None

    return {
        "rubric_id": payload.get("id", "custom_question"),
        "title": payload.get("question_si", "custom_question"),
        "max_score": sum(int(c.get("marks", 0)) for c in criteria),
        "selection_keywords": _extract_sinhala_terms(payload.get("question_si", "")),
        "criteria": criteria,
    }


def _contains_english(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text or ""))


def _normalize_relation_name(relation: str) -> str:
    return relation.replace(" ", "").strip()


def _extract_sinhala_terms(text: str) -> list[str]:
    terms = re.findall(r"[\u0D80-\u0DFF]{2,}", text)
    return list(dict.fromkeys(terms))


def _format_chunks_for_prompt(chunks: list[dict[str, Any]]) -> str:
    lines = []
    for chunk in chunks:
        meta = chunk["metadata"]
        short_text = chunk["content"][:700]
        lines.append(
            f"[chunk_id={meta['chunk_id']}, source={meta['source_file']}, topic={meta['topic_label']}]\n{short_text}"
        )
    return "\n\n".join(lines)


def _build_expanded_query(question: str, student_answer: str, marking_guide: str, rubric: dict[str, Any]) -> str:
    concepts = ONTOLOGY.get("concepts", {})
    entities = ONTOLOGY.get("entities", {})
    relationships = ONTOLOGY.get("relationships", {})

    expanded_terms: list[str] = ["වැව", "ජලාශ", "පාලනය"]

    for criterion in rubric.get("criteria", []):
        for concept_id in criterion.get("concept_ids", []):
            concept = concepts.get(concept_id, {})
            expanded_terms.extend(concept.get("answer_keywords", []))
            expanded_terms.extend(concept.get("evidence_keywords", []))

            for entity_ref in concept.get("entity_refs", []):
                entity = entities.get(entity_ref, {})
                expanded_terms.append(entity.get("name", ""))
                expanded_terms.extend(entity.get("synonyms", []))

            for rel_ref in concept.get("relationship_refs", []):
                rel = relationships.get(rel_ref, {})
                expanded_terms.append(rel.get("subject", ""))
                expanded_terms.append(rel.get("relation", ""))
                expanded_terms.append(rel.get("object", ""))
                expanded_terms.extend(rel.get("synonyms", []))

    # Topic-aware dynamic expansion.
    qtext = f"{question} {marking_guide}"
    if "වැව" in qtext or "ජල" in qtext or "වාරි" in qtext:
        for entity in entities.values():
            if entity.get("type") == "system" and ("ජල" in entity.get("name", "") or "වාරි" in " ".join(entity.get("synonyms", []))):
                expanded_terms.append(entity.get("name", ""))
                expanded_terms.extend(entity.get("synonyms", []))

    unique_terms = list(dict.fromkeys(term.strip() for term in expanded_terms if term and term.strip()))
    unique_terms = unique_terms[:40]

    return (
        f"{question}\n{student_answer}\n{marking_guide}\n"
        + " ".join(unique_terms)
    )


def _build_retrievers() -> tuple[Any, BM25Retriever]:
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 10})

    raw = vector_db.get(include=["documents", "metadatas"])
    docs = raw.get("documents") or []
    metas = raw.get("metadatas") or []
    bm25_docs: list[Document] = []
    for idx, content in enumerate(docs):
        if not content or not content.strip():
            continue
        meta = metas[idx] if idx < len(metas) and metas[idx] else {}
        bm25_docs.append(Document(page_content=content, metadata=meta))

    if not bm25_docs:
        raise ValueError("Vector DB හි documents හමු නොවීය. scripts/build_vector_db.py නැවත ධාවනය කරන්න.")

    bm25 = BM25Retriever.from_documents(bm25_docs)
    bm25.k = 10
    return vector_retriever, bm25


VECTOR_RETRIEVER, BM25 = _build_retrievers()


# Ontology utility functions requested in assignment.
def match_entity(answer: str) -> list[dict[str, Any]]:
    hits = []
    entities = ONTOLOGY.get("entities", {})
    for entity_id, entity in entities.items():
        names = [entity.get("name", "")] + entity.get("synonyms", [])
        matched = [name for name in names if name and name in answer]
        if matched:
            hits.append(
                {
                    "entity_id": entity_id,
                    "name": entity.get("name", ""),
                    "type": entity.get("type", "unknown"),
                    "matched_terms": matched,
                }
            )
    return hits


def check_relationship(entity: str, relation: str, evidence_text: str) -> list[dict[str, Any]]:
    matches = []
    relation_key = _normalize_relation_name(relation)

    for rel_id, rel in ONTOLOGY.get("relationships", {}).items():
        subject = rel.get("subject", "")
        rel_name = rel.get("relation", "")
        rel_name_key = _normalize_relation_name(rel_name)

        if entity and entity not in subject:
            continue
        if relation_key and relation_key not in rel_name_key:
            continue

        obj = rel.get("object", "")
        supported = subject in evidence_text and obj in evidence_text
        matches.append(
            {
                "relationship_id": rel_id,
                "subject": subject,
                "relation": rel_name,
                "object": obj,
                "supported_by_evidence": supported,
            }
        )

    return matches


def detect_concept(answer: str) -> list[dict[str, Any]]:
    found = []
    for concept_id, concept in ONTOLOGY.get("concepts", {}).items():
        keywords = concept.get("answer_keywords", [])
        matched = [kw for kw in keywords if kw in answer]
        if matched:
            found.append(
                {
                    "concept_id": concept_id,
                    "label": concept.get("label", concept_id),
                    "matched_terms": matched,
                }
            )
    return found


def _select_rubric(question: str, marking_guide: str) -> dict[str, Any]:
    text = f"{question} {marking_guide}"
    rubrics = ONTOLOGY.get("rubrics", [])
    if not rubrics:
        raise ValueError("Ontology rubrics හමු නොවීය.")

    selected = rubrics[0]
    best = -1
    for rubric in rubrics:
        score = sum(1 for kw in rubric.get("selection_keywords", []) if kw in text)
        if score > best:
            best = score
            selected = rubric
    return selected


def _resolve_rubric(question: str, marking_guide: str) -> dict[str, Any]:
    custom = _extract_custom_rubric(marking_guide)
    if custom:
        return custom
    return _select_rubric(question, marking_guide)


def _build_ontology_match_output(state: GradeState) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    answer = state.get("student_answer", "")
    fallback_chunk = (
        state["retrieved_chunks"][0]["metadata"].get("chunk_id", "නොමැත")
        if state.get("retrieved_chunks")
        else "නොමැත"
    )

    for rel in state.get("relationship_checks", []):
        expected = rel.get("object", "")
        student_has = expected in answer if expected else False
        evidence_has = bool(rel.get("supported_by_evidence", False))

        if student_has and evidence_has:
            label = "✔ සම්පූර්ණ ගැළපීම"
        elif student_has or evidence_has:
            label = "~ අර්ධ ගැළපීම"
        else:
            label = "✘ නොගැළපීම"

        out.append(
            {
                "subject": rel.get("subject", ""),
                "relation": rel.get("relation", ""),
                "expected": expected,
                "student": expected if student_has else "නොමැත",
                "label": label,
                "chunk_id": fallback_chunk,
            }
        )

    return out


def _build_missing_concepts(state: GradeState) -> list[str]:
    missing: list[str] = []

    for item in state.get("coverage_output", {}).get("items", []):
        status = item.get("status", "not_satisfied")
        if status != "fully_satisfied":
            missing.append(f"මාපකය අසම්පූර්ණයි: {item.get('criterion', 'නොදනී')}")

    concept_map = ONTOLOGY.get("concepts", {})
    answer = state.get("student_answer", "")
    rubric = state.get("selected_rubric", {})
    required_ids: set[str] = set()
    for criterion in rubric.get("criteria", []):
        for cid in criterion.get("concept_ids", []):
            if cid:
                required_ids.add(cid)

    for cid in sorted(required_ids):
        concept = concept_map.get(cid, {})
        kws = concept.get("answer_keywords", [])
        if kws and not any(kw in answer for kw in kws):
            missing.append(f"ඔන්ටොලොජි කරුණ නොමැත: {concept.get('label', cid)}")

    return list(dict.fromkeys(missing))


def _build_final_justification(state: GradeState, grounded: int, total: int) -> str:
    score_out = state["scoring_output"]
    confidence = float(score_out.get("retrieval_confidence", 0.0))
    chunk_hint = (
        state["retrieved_chunks"][0]["metadata"].get("chunk_id", "නොමැත")
        if state.get("retrieved_chunks")
        else "නොමැත"
    )

    lines = [
        f"ලකුණු: {score_out['final_score']}/20",
        f"(සංශෝධිත ලකුණු: {score_out['confidence_adjusted_score']}/20)",
        "",
        "විස්තරය:",
    ]

    for idx, item in enumerate(score_out.get("breakdown", []), start=1):
        evidence_chunk = item.get("evidence_chunk_ids", [chunk_hint])
        cid = evidence_chunk[0] if evidence_chunk else chunk_hint
        quote = "පාඨපොත් සාක්ෂියක් සනාථ විය."
        for ch in state.get("retrieved_chunks", []):
            if ch["metadata"].get("chunk_id") == cid:
                quote = ch["content"].strip().replace("\n", " ")[:120]
                break

        lines.append(f"* මාපක {idx}: {item['awarded']}/{item['max_marks']} (chunk: {cid})")
        lines.append(f"  හේතුව: {item['reason']} (chunk: {cid})")
        lines.append(f"  සාක්ෂි: \"{quote}\" (chunk: {cid})")

    lines.append("")
    lines.append("ඔන්ටොලොජි සම්බන්ධය:")
    ontology_items = state.get("ontology_match_output", [])
    if ontology_items:
        for rel in ontology_items:
            lines.append(
                f"* {rel['subject']} -> {rel['relation']} -> {rel['expected']} -> {rel['student']} -> {rel['label']} (chunk: {rel['chunk_id']})"
            )
    else:
        lines.append(f"* සනාථ කළ හැකි සම්බන්ධතාවක් හමු නොවීය. (chunk: {chunk_hint})")

    lines.append("")
    lines.append("අහිමි කරුණු:")
    missing = state.get("missing_concepts", [])
    if missing:
        for item in missing:
            lines.append(f"* {item} (chunk: {chunk_hint})")
    else:
        lines.append(f"* අහිමි කරුණු හමු නොවීය. (chunk: {chunk_hint})")

    lines.append("")
    lines.append("සාක්ෂි ආවරණය:")
    lines.append(f"{grounded}/{total} වාක්‍ය පාඨපොත් සාක්ෂි සමඟ සම්බන්ධයි (chunk: {chunk_hint})")

    lines.append("")
    lines.append("විශ්වාස මට්ටම:")
    lines.append(f"* retrieval confidence = {confidence:.4f} (chunk: {chunk_hint})")
    if state.get("low_confidence"):
        lines.append(
            f"* retrieval විශ්වාස මට්ටම ({confidence:.4f}) අඩු බැවින්, ලබාදුන් සාක්ෂි සම්පූර්ණ නොවිය හැකි බැවින් ලකුණු අඩු කරන ලදී. (chunk: {chunk_hint})"
        )

    lines.append("")
    lines.append("සටහන:")
    lines.append(f"සටහන: මෙම ප්‍රතිඵල පාඨපොත් දත්ත සහ නියමිත නීති මත පදනම්ව නිර්මාණය කර ඇත. (chunk: {chunk_hint})")

    return "\n".join(lines)


def _hybrid_retrieve(query: str, top_k: int) -> tuple[list[dict[str, Any]], float]:
    vector_docs = VECTOR_RETRIEVER.invoke(query)
    bm25_docs = BM25.invoke(query)

    terms = _extract_sinhala_terms(query)
    keyword_weight = float(POLICY.get("keyword_boost_weight", 0.12))

    merged: dict[str, dict[str, Any]] = {}
    sources = [(vector_docs, 0.65), (bm25_docs, 0.35)]

    for docs, base_weight in sources:
        for rank, doc in enumerate(docs):
            key = doc.metadata.get("chunk_id") or doc.page_content[:80]
            score = base_weight * (1.0 / (rank + 1))
            overlap = sum(1 for term in terms if term in doc.page_content)
            # Keep keyword boost bounded so rank quality is stable.
            normalized_boost = min(keyword_weight, keyword_weight * (overlap / 4.0))
            score += normalized_boost

            if key not in merged:
                merged[key] = {
                    "content": doc.page_content,
                    "metadata": {
                        "source_file": doc.metadata.get("source_file", "unknown"),
                        "chunk_id": doc.metadata.get("chunk_id", "unknown"),
                        "topic_label": doc.metadata.get("topic_label", "සාමාන්‍ය ඉතිහාසය"),
                    },
                    "score": 0.0,
                }
            merged[key]["score"] += score

    ranked = sorted(merged.values(), key=lambda item: item["score"], reverse=True)
    top = ranked[:top_k]
    confidence = sum(item["score"] for item in top) / max(1, len(top))
    # Normalize confidence to [0, 1] range (BM25 scores can exceed 1.0)
    confidence = min(1.0, confidence)
    return top, confidence


def retrieval_agent(state: GradeState, status_callback: callable = None) -> GradeState:
    if status_callback:
        status_callback("Retrieval Agent", "Running retrieval...", "⏳")
    print("\n[Agent 1] Retrieval Agent running...")
    top_k = int(POLICY.get("top_k", 6))
    rubric = _resolve_rubric(state["question"], state["marking_guide"])
    state["selected_rubric"] = rubric
    query = _build_expanded_query(
        question=state["question"],
        student_answer=state["student_answer"],
        marking_guide=state["marking_guide"],
        rubric=rubric,
    )
    chunks, confidence = _hybrid_retrieve(query, top_k=top_k)

    threshold = float(POLICY.get("confidence_threshold", 0.4))
    state["retrieved_chunks"] = chunks
    state["retrieval_confidence"] = round(confidence, 4)
    state["low_confidence"] = len(chunks) == 0 or confidence < threshold
    print(f"[Agent 1] Retrieved {len(chunks)} chunks, confidence={state['retrieval_confidence']}")
    if status_callback:
        status_callback("Retrieval Agent", f"✓ Retrieved {len(chunks)} chunks (confidence={state['retrieval_confidence']:.4f})", "✅")
    return state


def coverage_checker_agent(state: GradeState, status_callback: callable = None) -> GradeState:
    if status_callback:
        status_callback("Coverage Checker", "Checking rubric coverage...", "⏳")
    print("[Agent 2] Coverage Checker Agent running...")

    rubric = state.get("selected_rubric") or _resolve_rubric(state["question"], state["marking_guide"])
    state["selected_rubric"] = rubric

    answer = state["student_answer"]
    evidence_text = "\n".join(chunk["content"] for chunk in state["retrieved_chunks"])

    state["entity_matches"] = match_entity(answer)
    state["relationship_checks"] = []
    for hit in state["entity_matches"]:
        state["relationship_checks"].extend(check_relationship(hit["name"], "", evidence_text))

    concept_hits = detect_concept(answer)
    concept_ids = {item["concept_id"] for item in concept_hits}
    concept_map = ONTOLOGY.get("concepts", {})

    criteria_input = []
    deterministic_items = []

    for criterion in rubric.get("criteria", []):
        criterion_id = criterion.get("criterion_id", "unknown")
        concept_refs = criterion.get("concept_ids", [])
        expected_terms = criterion.get("expected_terms", [])

        if expected_terms:
            matched_terms = [term for term in expected_terms if term and term in answer]
            term_ratio = len(matched_terms) / max(1, len(expected_terms))
            evidence_chunks = [
                ch for ch in state["retrieved_chunks"]
                if any(term in ch["content"] for term in expected_terms)
            ]
            evidence_refs = [ch["metadata"]["chunk_id"] for ch in evidence_chunks[:2]]

            if term_ratio >= 0.99 and evidence_chunks:
                status = "fully_satisfied"
            elif term_ratio > 0.0:
                status = "partially_satisfied"
            else:
                status = "not_satisfied"

            deterministic_items.append(
                {
                    "criterion_id": criterion_id,
                    "criterion": criterion.get("description", ""),
                    "max_marks": int(criterion.get("marks", 0)),
                    "status": status,
                    "ratio": term_ratio,
                    "evidence_chunk_ids": list(dict.fromkeys(evidence_refs)),
                }
            )

            criteria_input.append(
                {
                    "criterion_id": criterion_id,
                    "criterion": criterion.get("description", ""),
                    "max_marks": int(criterion.get("marks", 0)),
                }
            )
            continue

        evidence_refs: list[str] = []
        status_scores = []

        for concept_id in concept_refs:
            concept = concept_map.get(concept_id, {})
            in_answer = concept_id in concept_ids

            evidence_chunks = [
                ch for ch in state["retrieved_chunks"]
                if any(kw in ch["content"] for kw in concept.get("evidence_keywords", []))
            ]
            in_evidence = len(evidence_chunks) > 0

            rel_refs = concept.get("relationship_refs", [])
            rel_ok = True
            if rel_refs:
                rel_ok = all(
                    any(
                        rc["relationship_id"] == rel_ref and rc["supported_by_evidence"]
                        for rc in state["relationship_checks"]
                    )
                    for rel_ref in rel_refs
                )

            if in_answer and in_evidence and rel_ok:
                status_scores.append(1.0)
            elif in_answer and (in_evidence or rel_ok):
                status_scores.append(0.5)
            else:
                status_scores.append(0.0)

            evidence_refs.extend(ch["metadata"]["chunk_id"] for ch in evidence_chunks[:2])

        avg_ratio = sum(status_scores) / max(1, len(status_scores))
        status = "not_satisfied"
        if avg_ratio >= 0.99:
            status = "fully_satisfied"
        elif avg_ratio > 0.0:
            status = "partially_satisfied"

        deterministic_items.append(
            {
                "criterion_id": criterion_id,
                "criterion": criterion.get("description", ""),
                "max_marks": int(criterion.get("marks", 0)),
                "status": status,
                "ratio": avg_ratio,
                "evidence_chunk_ids": list(dict.fromkeys(evidence_refs)),
            }
        )

        criteria_input.append(
            {
                "criterion_id": criterion_id,
                "criterion": criterion.get("description", ""),
                "max_marks": int(criterion.get("marks", 0)),
            }
        )

    coverage_prompt = ChatPromptTemplate.from_template(
        """
ඔබ ඉතිහාස විෂය ගුරුවරයෙකි.

පහත දත්ත භාවිතා කරමින්, ශිෂ්‍ය පිළිතුර විශ්ලේෂණය කරන්න:

ප්‍රශ්නය:
{question}

ශිෂ්‍ය පිළිතුර:
{answer}

ලකුණු මාර්ගෝපදේශය:
{criteria}

පාඨපොත් දත්ත:
{retrieved_chunks}

ඔන්ටොලොජි සම්බන්ධතා:
{ontology_relations}

ඔබ කළ යුතු දේ:
1. සෑම ලකුණු මාපකයක් සඳහා fully_satisfied / partially_satisfied / not_satisfied යන්න සලකන්න.
2. පිළිතුර පාඨපොත් දත්ත සමඟ ගැළපේද පරීක්ෂා කරන්න.
3. ඔන්ටොලොජි සම්බන්ධතා සත්‍යද පරීක්ෂා කරන්න.

JSON ලෙස පමණක් පිළිතුරු දෙන්න:
{{
  "items": [
    {{
      "criterion_id": "...",
      "status": "fully_satisfied|partially_satisfied|not_satisfied",
      "reason": "...",
      "evidence_chunk_ids": ["..."]
    }}
  ]
}}
"""
    )

    llm_response = (coverage_prompt | llama).invoke(
        {
            "question": state["question"],
            "answer": answer,
            "criteria": json.dumps(criteria_input, ensure_ascii=False, indent=2),
            "retrieved_chunks": _format_chunks_for_prompt(state["retrieved_chunks"]),
            "ontology_relations": json.dumps(state["relationship_checks"], ensure_ascii=False, indent=2),
        }
    )

    llm_json = _safe_json_extract(str(llm_response.content))
    llm_map = {it.get("criterion_id"): it for it in llm_json.get("items", []) if isinstance(it, dict)}

    merged_items = []
    for item in deterministic_items:
        llm_item = llm_map.get(item["criterion_id"], {})
        merged_items.append(
            {
                "criterion_id": item["criterion_id"],
                "criterion": item["criterion"],
                "max_marks": item["max_marks"],
                "status": item["status"],
                "ratio": item["ratio"],
                "reason": llm_item.get("reason", "පාඨපොත් දත්ත සහ ඔන්ටොලොජි සම්බන්ධතා මත තීරණය කරන ලදී."),
                "evidence_chunk_ids": llm_item.get("evidence_chunk_ids", item["evidence_chunk_ids"]),
            }
        )

    state["coverage_output"] = {
        "rubric_id": rubric.get("rubric_id", "unknown"),
        "items": merged_items,
    }
    print(f"[Agent 2] Coverage completed for {len(merged_items)} criteria.")
    if status_callback:
        status_callback("Coverage Checker", f"✓ Evaluated {len(merged_items)} criteria", "✅")
    return state


def scoring_agent(state: GradeState, status_callback: callable = None) -> GradeState:
    if status_callback:
        status_callback("Scoring Agent", "Computing criterion marks...", "⏳")
    print("[Agent 3] Scoring Agent running...")
    coverage = state["coverage_output"]

    scoring_prompt = ChatPromptTemplate.from_template(
        """
ඔබ විභාග පරීක්ෂකයෙකි.

පහත විශ්ලේෂණය මත ලකුණු ලබා දෙන්න:

{coverage_output}

නීති:
* සම්පූර්ණ පිළිතුර -> සම්පූර්ණ ලකුණු
* අර්ධ පිළිතුර -> අර්ධ ලකුණු (50%)
* වැරදි / නොමැති -> 0
* පාඨපොත් සහාය නොමැති නම් -> ලකුණු අඩු කරන්න

JSON ලෙස:
{{
  "final_score": 0,
  "breakdown": [
    {{"criterion_id":"...","awarded":0,"reason":"..."}}
  ]
}}
"""
    )

    llm_response = (scoring_prompt | llama).invoke(
        {
            "coverage_output": json.dumps(coverage, ensure_ascii=False, indent=2),
        }
    )
    llm_json = _safe_json_extract(str(llm_response.content))

    status_ratio = {
        "fully_satisfied": 1.0,
        "partially_satisfied": 0.5,
        "not_satisfied": 0.0,
    }

    deterministic_breakdown = []
    deterministic_score = 0
    for item in coverage.get("items", []):
        ratio = float(item.get("ratio", status_ratio.get(item.get("status", "not_satisfied"), 0.0)))
        ratio = max(0.0, min(1.0, ratio))
        marks = int(item.get("max_marks", 0))
        awarded = int(round(marks * float(ratio)))

        deterministic_score += awarded
        deterministic_breakdown.append(
            {
                "criterion_id": item.get("criterion_id", "unknown"),
                "criterion": item.get("criterion", ""),
                "max_marks": marks,
                "awarded": awarded,
                "reason": item.get("reason", ""),
                "evidence_chunk_ids": item.get("evidence_chunk_ids", []),
            }
        )

    # Deterministic evidence-grounded score is authoritative.
    final_score = max(0, min(20, deterministic_score))

    conf_threshold = float(POLICY.get("confidence_threshold", 0.4))
    confidence = float(state["retrieval_confidence"])
    penalty = 0.0
    if confidence < conf_threshold:
        penalty = (conf_threshold - confidence) * 5.0
    confidence_adjusted = max(0.0, float(final_score) - penalty)
    confidence_adjusted = round(confidence_adjusted, 1)

    state["final_score"] = final_score
    state["confidence_adjusted_score"] = confidence_adjusted
    state["scoring_output"] = {
        "final_score": final_score,
        "confidence_adjusted_score": confidence_adjusted,
        "retrieval_confidence": state["retrieval_confidence"],
        "confidence_penalty": round(penalty, 1),
        "low_confidence": state["low_confidence"],
        "breakdown": deterministic_breakdown,
    }
    print(f"[Agent 3] Score={final_score}/20, confidence_adjusted_score={confidence_adjusted}/20")
    if status_callback:
        status_callback("Scoring Agent", f"✓ Score: {final_score}/20 (Adjusted: {confidence_adjusted}/20)", "✅")
    return state


def _fallback_explanation(state: GradeState) -> str:
    score_out = state["scoring_output"]
    lines = [
        f"ලකුණු: {score_out['final_score']}/20",
        f"(සංශෝධිත ලකුණු: {score_out['confidence_adjusted_score']}/20)",
        "",
        "විස්තරය:",
    ]

    for item in score_out.get("breakdown", []):
        lines.append(f"* මාපක: {item['criterion']} ({item['awarded']}/{item['max_marks']})")
        lines.append("  හේතුව:")
        lines.append(f"  {item['reason']}")
        lines.append("  සාක්ෂි:")
        if item.get("evidence_chunk_ids"):
            cid = item["evidence_chunk_ids"][0]
            lines.append(f"  \"පාඨපොත් සනාථය\" (chunk: {cid})")
        else:
            lines.append("  \"සෘජු පාඨපොත් සාක්ෂි නොමැත\" (chunk: නොමැත)")
        lines.append("")

    lines.append("ඔන්ටොලොජි සම්බන්ධය:")
    for rel in state.get("relationship_checks", []):
        expected = rel.get("object", "")
        student_match = expected if expected and expected in state.get("student_answer", "") else "නොමැත"
        mark = "✔" if student_match != "නොමැත" else "✘"
        lines.append(
            f"* {rel.get('subject','')} -> {rel.get('relation','')} -> {expected} -> {student_match} -> {mark} (chunk: {state['retrieved_chunks'][0]['metadata']['chunk_id'] if state.get('retrieved_chunks') else 'නොමැත'})"
        )

    lines.append("විශ්වාස මට්ටම:")
    lines.append(
        f"* retrieval confidence = {score_out['retrieval_confidence']:.2f} (chunk: {state['retrieved_chunks'][0]['metadata']['chunk_id'] if state.get('retrieved_chunks') else 'නොමැත'})"
    )

    if state["low_confidence"]:
        lines.append(
            f"* විශ්වාස මට්ටම අඩු බැවින් ({score_out['retrieval_confidence']:.2f}), ලකුණු {score_out['confidence_penalty']:.1f} කින් අඩු කරන ලදී. (chunk: {state['retrieved_chunks'][0]['metadata']['chunk_id'] if state.get('retrieved_chunks') else 'නොමැත'})"
        )

    return "\n".join(lines)


def enforce_evidence(explanation: str) -> tuple[str, int, int]:
    sentences = explanation.split("\n")
    validated = []
    grounded = 0
    total = 0
    structural_headers = {
        "ලකුණු:",
        "(සංශෝධිත ලකුණු:",
        "විස්තරය:",
        "ඔන්ටොලොජි සම්බන්ධය:",
        "අහිමි කරුණු:",
        "සාක්ෂි ආවරණය:",
        "විශ්වාස මට්ටම:",
        "සටහන:",
    }
    for s in sentences:
        if not s.strip():
            continue
        total += 1
        if any(s.strip().startswith(header) for header in structural_headers):
            validated.append(s)
            continue
        if "chunk:" in s or "chunks:" in s:
            grounded += 1
            validated.append(s)
        else:
            validated.append(f"[UNVERIFIED] {s}")

    print(f"[Evidence Check] {grounded}/{total} sentences grounded")
    return "\n".join(validated), grounded, total


def explanation_agent(state: GradeState, status_callback: callable = None) -> GradeState:
    if status_callback:
        status_callback("Explanation Agent", "Generating explanation...", "⏳")
    print("[Agent 4] Explanation Agent running...")

    ontology_lines = []
    for rel in state.get("relationship_checks", []):
        expected = rel.get("object", "")
        student_value = expected if expected and expected in state.get("student_answer", "") else "නොගැලපේ"
        mark = "✔" if student_value != "නොගැලපේ" else "✘"
        ontology_lines.append(
            f"{rel.get('subject','')} -> {rel.get('relation','')} -> {expected} -> {student_value} -> {mark}"
        )
    if not ontology_lines:
        ontology_lines.append("ඔන්ටොලොජි සම්බන්ධතා සනාථ කිරීමට ප්‍රමාණවත් entity හමු නොවීය.")

    explanation_prompt = ChatPromptTemplate.from_template(
        """
ඔබ ඉතා නිවැරදි සහ සාක්ෂි-පදනම් ශ්‍රේණිගත කිරීම සිදු කරන ඉතිහාස ගුරුවරයෙකි.

පහත දත්ත භාවිතා කරමින්, පැහැදිලි ශ්‍රේණිගත කිරීමේ විස්තරයක් සකස් කරන්න:

{scoring_output}
{retrieved_chunks}
{ontology_relations}

අනිවාර්ය නීති:
1. සෑම ලකුණු මාපකයක් සඳහා:
   * ලැබුණු ලකුණු
   * හේතුව
   * පාඨපොත් සාක්ෂි (quote එකක් + chunk id)
2. සෑම වාක්‍යයක්ම chunk id එකකට සම්බන්ධ විය යුතුය.
3. ඔන්ටොලොජි සම්බන්ධතා පහත ආකාරයෙන් දක්වන්න:
   Entity -> Relation -> Expected -> Student Answer Match (✔ / ✘)
4. සම්පූර්ණ ලකුණු ලබා දුන් විටද හේතුව පැහැදිලි කරන්න.
5. retrieval confidence අඩු නම් එය සඳහන් කර ලකුණු අඩු කිරීම පැහැදිලි කරන්න.
6. අවසානයේ සාරාංශයක් දක්වන්න.
7. සම්පූර්ණ පිළිතුර සිංහලෙන් ලබා දෙන්න.

අනිවාර්ය ප්‍රතිදාන ආකෘතිය:
ලකුණු: X/20
(සංශෝධිත ලකුණු: Y/20)

විස්තරය:
* මාපක 1: X/X
  හේතුව:
  සාක්ෂි: "..." (chunk: ...)

* මාපක 2: ...

ඔන්ටොලොජි සම්බන්ධය:
... (chunk: ...)

විශ්වාස මට්ටම:
... (chunk: ...)

සාරාංශය:
... (chunk: ...)

භාවිත කළ පාඨපොත්:
* chunk id + source (chunks: ...)
"""
    )

    llm_response = (explanation_prompt | llama).invoke(
        {
            "scoring_output": json.dumps(state["scoring_output"], ensure_ascii=False, indent=2),
            "retrieved_chunks": _format_chunks_for_prompt(state["retrieved_chunks"]),
            "ontology_relations": "\n".join(ontology_lines),
        }
    )
    text = str(llm_response.content).strip()

    if _contains_english(text) or "ලකුණු:" not in text or "භාවිත කළ පාඨපොත්:" not in text:
        text = _fallback_explanation(state)

    state["justification"] = text
    if status_callback:
        status_callback("Explanation Agent", "✓ Explanation generated", "✅")
    return state


def evidence_enforcement_agent(state: GradeState, status_callback: callable = None) -> GradeState:
    if status_callback:
        status_callback("Evidence Enforcement", "Validating grounding...", "⏳")
    print("[Agent 5] Evidence Enforcement Layer running...")
    state["ontology_match_output"] = _build_ontology_match_output(state)
    state["missing_concepts"] = _build_missing_concepts(state)

    # First-pass validation over model output for traceability.
    _, grounded, total = enforce_evidence(state.get("justification", ""))

    # Deterministic final Sinhala report with mandatory sections.
    final_text = _build_final_justification(state, grounded, total)
    validated, final_grounded, final_total = enforce_evidence(final_text)
    state["evidence_coverage"] = f"{final_grounded}/{final_total} වාක්‍ය පාඨපොත් සාක්ෂි සමඟ සම්බන්ධයි"
    state["justification"] = validated
    coverage_pct = (final_grounded / final_total * 100) if final_total > 0 else 0
    print(f"[Agent 5] Evidence enforcement completed. Coverage: {coverage_pct:.2f}%")
    if status_callback:
        status_callback("Evidence Enforcement", f"✓ Grounding verified ({coverage_pct:.1f}% coverage)", "✅")
    return state


def build_grading_graph():
    workflow = StateGraph(GradeState)
    workflow.add_node("retrieve", retrieval_agent)
    workflow.add_node("coverage", coverage_checker_agent)
    workflow.add_node("score", scoring_agent)
    workflow.add_node("explain", explanation_agent)
    workflow.add_node("evidence", evidence_enforcement_agent)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "coverage")
    workflow.add_edge("coverage", "score")
    workflow.add_edge("score", "explain")
    workflow.add_edge("explain", "evidence")
    workflow.add_edge("evidence", END)
    return workflow.compile()


def run_single_case(
    question: str,
    student_answer: str,
    marking_guide: str,
    status_callback: callable = None,
) -> dict[str, Any]:
    """Run the 5-agent grading pipeline with optional status callback."""
    state = GradeState(
        question=question,
        student_answer=student_answer,
        marking_guide=marking_guide,
        retrieved_chunks=[],
        retrieval_confidence=0.0,
        low_confidence=False,
        selected_rubric={},
        entity_matches=[],
        relationship_checks=[],
        coverage_output={},
        scoring_output={},
        final_score=0,
        confidence_adjusted_score=0,
        missing_concepts=[],
        ontology_match_output=[],
        evidence_coverage="0/0 වාක්‍ය පාඨපොත් සාක්ෂි සමඟ සම්බන්ධයි",
        justification="",
    )
    
    # Sequential agent execution with status callbacks
    state = retrieval_agent(state, status_callback)
    state = coverage_checker_agent(state, status_callback)
    state = scoring_agent(state, status_callback)
    state = explanation_agent(state, status_callback)
    state = evidence_enforcement_agent(state, status_callback)
    
    return {
        "final_score": state["final_score"],
        "confidence_adjusted_score": state["confidence_adjusted_score"],
        "retrieval_confidence": state["retrieval_confidence"],
        "evidence_coverage": state["evidence_coverage"],
        "scoring_output": state["scoring_output"],
        "ontology_match_output": state["ontology_match_output"],
        "missing_concepts": state["missing_concepts"],
        "justification": state["justification"],
    }


if __name__ == "__main__":
    sample = run_single_case(
        question="පණ්ඩුකාභය රජතුමාගේ දෙමව්පියන් කවුද?",
        student_answer="පණ්ඩුකාභයගේ පියා දීඝගාමිණී සහ මව චිත්‍රා ය.",
        marking_guide="රජුවරයා, පියා, මව නිවැරදිව තිබේ නම් ලකුණු 20 දක්වා දෙන්න.",
    )
    print(sample["justification"])
