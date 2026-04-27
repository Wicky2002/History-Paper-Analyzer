import os
import re
import shutil
import unicodedata
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
DB_DIR = os.path.join(BASE_DIR, "data", "vector_db")

CHUNK_MIN_WORDS = 260
CHUNK_MAX_WORDS = 400
CHUNK_TARGET_WORDS = 340
CHUNK_OVERLAP_WORDS = 80

TOPIC_KEYWORDS = {
    "රජවරු සහ පාලනය": ["රජ", "කුමරු", "පාලනය", "අනුරාධපුර"],
    "ජල තාක්ෂණය සහ වැව්": ["වැව", "අභය", "බසවක්කුලම", "ඇළ", "වාරි"],
    "ගොවිතැන සහ ආර්ථිකය": ["ගොවිතැන", "හේන්", "කුඹුරු", "කෘෂිකර්මාන්ත"],
}


def normalize_sinhala_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)

    # Common OCR/FM conversion artifacts.
    text = text.replace("`", "")
    text = text.replace("˘", "")
    text = text.replace("\u200b", "")

    noise_line_pattern = re.compile(r"නොමිලේ\s*බෙදා\s*හැරීම\s*සඳහා", re.IGNORECASE)
    cleaned_lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            cleaned_lines.append("")
            continue
        if noise_line_pattern.search(line):
            continue
        if line.isdigit():
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def infer_topic_label(text: str) -> str:
    best_topic = "සාමාන්‍ය ඉතිහාසය"
    best_score = 0
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > best_score:
            best_score = score
            best_topic = topic
    return best_topic


def split_long_paragraph(paragraph: str, max_words: int) -> list[str]:
    words = paragraph.split()
    if len(words) <= max_words:
        return [paragraph]

    # Prefer sentence-ish boundaries first for coherence.
    parts = re.split(r"(?<=[\.\?!])\s+", paragraph)
    if len(parts) == 1:
        slices = []
        for i in range(0, len(words), max_words):
            slices.append(" ".join(words[i : i + max_words]))
        return slices

    out = []
    buf = []
    for part in parts:
        part_words = part.split()
        if len(buf) + len(part_words) <= max_words:
            buf.extend(part_words)
        else:
            if buf:
                out.append(" ".join(buf))
            if len(part_words) > max_words:
                out.extend(split_long_paragraph(" ".join(part_words), max_words))
                buf = []
            else:
                buf = part_words
    if buf:
        out.append(" ".join(buf))
    return out


def chunk_by_words(text: str, min_words: int, max_words: int, overlap_words: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units = []
    for paragraph in paragraphs:
        units.extend(split_long_paragraph(paragraph, max_words))

    chunks = []
    current_words = []

    for unit in units:
        unit_words = unit.split()
        if not current_words:
            current_words = unit_words.copy()
            continue

        if len(current_words) + len(unit_words) <= max_words:
            current_words.extend(unit_words)
        else:
            chunks.append(" ".join(current_words).strip())
            overlap = current_words[-overlap_words:] if overlap_words < len(current_words) else current_words
            current_words = overlap + unit_words

            if len(current_words) > max_words:
                chunks.append(" ".join(current_words[:max_words]).strip())
                overlap = current_words[max_words - overlap_words : max_words]
                current_words = overlap + current_words[max_words:]

    if current_words:
        chunks.append(" ".join(current_words).strip())

    # Merge very small trailing chunks to preserve ~300-500 range where possible.
    merged = []
    for chunk in chunks:
        if not merged:
            merged.append(chunk)
            continue
        if len(chunk.split()) < min_words and len(merged[-1].split()) < max_words:
            candidate = merged[-1] + " " + chunk
            if len(candidate.split()) <= (max_words + 80):
                merged[-1] = candidate.strip()
                continue
        merged.append(chunk)

    return merged


def load_documents() -> list[Document]:
    docs = []
    processed_path = Path(PROCESSED_DIR)
    for txt_file in sorted(processed_path.glob("*.txt")):
        raw_text = txt_file.read_text(encoding="utf-8")
        cleaned = normalize_sinhala_text(raw_text)
        chunks = chunk_by_words(cleaned, CHUNK_MIN_WORDS, CHUNK_MAX_WORDS, CHUNK_OVERLAP_WORDS)

        for idx, chunk_text in enumerate(chunks, start=1):
            topic = infer_topic_label(chunk_text)
            chunk_id = f"{txt_file.stem.replace(' ', '_').lower()}_{idx:03d}"
            docs.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "source_file": txt_file.name,
                        "chunk_id": chunk_id,
                        "topic_label": topic,
                        "word_count": len(chunk_text.split()),
                    },
                )
            )
    return docs


def main():
    print("Loading and cleaning Sinhala textbook text files...")
    documents = load_documents()
    if not documents:
        raise ValueError("data/processed තුළ .txt ගොනු හමු නොවීය.")

    print(f"Prepared {len(documents)} semantically coherent chunks.")
    avg_words = sum(doc.metadata["word_count"] for doc in documents) / len(documents)
    print(f"Average chunk length: {avg_words:.1f} words")

    print("Initializing offline embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

    print("Writing local Chroma vector store...")
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name="history_processed_si",
    )

    print(f"Vector DB rebuilt successfully at: {DB_DIR}")


if __name__ == "__main__":
    main()