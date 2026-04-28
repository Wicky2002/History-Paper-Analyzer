History-Paper-Analyzer
======================

Offline Intelligent Sinhala Open-Ended Answer Scorer (Anuradhapura Period)

A local, explainable pipeline for grading Sinhala open-ended history answers using:
- Local LLM (Ollama / llama3) for deterministic explanations
- Vector retrieval (Chroma + BM25) for textbook grounding
- Ontology-based reasoning for entity/relation checks
- Rubric-driven proportional scoring with evidence enforcement

Key features
- Rubric-aware partial-credit scoring
- Evidence grounding: links back to textbook chunks
- Ontology entity-relation verification and missing-concept detection
- Streamlit-based UI for interactive grading and live workflow visualization
- Fully offline capable (no cloud APIs required for core pipeline)

Repository structure (high level)
- [agents/workflow.py](agents/workflow.py) — core 5-agent pipeline (retrieval, coverage, scoring, explanation, evidence enforcement)
- [streamlit_app.py](streamlit_app.py) — interactive UI for selecting questions and running evaluations
- [data/question_bank_si.json](data/question_bank_si.json) — question bank with rubrics and demo answers
- [data/ontology/history_ontology_si.json](data/ontology/history_ontology_si.json) — domain ontology
- [data/processed/](data/processed/) — generated outputs (evaluation JSON, processed text)
- [scripts/evaluate_demo_set.py](scripts/evaluate_demo_set.py) — batch evaluation harness producing metrics
- [scripts/process_pdfs.py](scripts/process_pdfs.py) & [scripts/build_vector_db.py](scripts/build_vector_db.py) — helpers for building the vector DB

Quick start (Windows)
1. Create and activate the virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies

```powershell
pip install -r requirements.txt
```

3. (Optional) Install extra LangChain adapters if you see deprecation warnings:

```powershell
pip install langchain-core langchain-ollama langchain-chroma
```

4. Run the Streamlit UI (use the venv Python to ensure the right env)

```powershell
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

5. Run the demo evaluation harness

```powershell
.venv\Scripts\python.exe scripts/evaluate_demo_set.py
```

Developer test

```powershell
.venv\Scripts\python.exe test_fixes.py
```

Notes & troubleshooting
- If you see ModuleNotFoundError for `langchain_core` or similar, install the LangChain adapters as above. Some package names changed across versions; prefer `langchain-core`, `langchain-ollama`, `langchain-chroma` if required.
- If transformers triggers optional image/video model imports (torchvision), and you don't need them, you can either install `torchvision` or constrain `transformers` to a version that avoids eager imports. To install torchvision (CPU-only wheel), use your platform-specific instructions or conda/pip: `pip install torchvision`.
- The pipeline uses an offline Embedding/LLM; ensure your local model runtime (Ollama or similar) is running if required by your configuration.

Design notes
- The scoring pipeline is intentionally hybrid: deterministic keyword/ontology checks + LLM prompts for human-friendly explanations. This improves explainability and reduces arbitrary over-scoring.
- Retrieval confidence is capped to 1.0 and a confidence penalty mechanism is implemented to flag low-confidence grading decisions.

Files to inspect for customization
- `agents/workflow.py` — tune `POLICY` values (top_k, confidence_threshold) and the explanation prompts.
- `data/question_bank_si.json` — add or edit questions, marking schemes, and sample answers.
- `data/vector_db/` — contains Chroma DB files; recreate with `scripts/build_vector_db.py` when adding/updating source documents.

How to run a single case in Python (example)

```python
from agents.workflow import run_single_case
import json

with open('data/question_bank_si.json', 'r', encoding='utf-8') as f:
    q = json.load(f)[0]

result = run_single_case(
    question=q['question_si'],
    student_answer=q['sample_answers']['correct_100'],
    marking_guide=json.dumps(q, ensure_ascii=False)
)
print(result['final_score'], result['retrieval_confidence'])
```

Contributing
- Open an issue for bugs or feature requests.
- Use small, focused PRs; add tests where appropriate.

License
- (Add your preferred license here.)

Contact
- For questions about this repo, contact the maintainer in the project metadata or README updates.
