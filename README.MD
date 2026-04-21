# Offline Intelligent Sinhala Open-Ended Answer Scorer 🇱🇰

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Ollama](https://img.shields.io/badge/Ollama-Offline-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![NLP](https://img.shields.io/badge/NLP-Sinhala-green)

## 📌 Project Overview
This project is an offline, AI-driven grading system designed to evaluate Sinhala open-ended answers related to the History of Sri Lanka (CS4032 Natural Language Processing). Utilizing a fully local LLM via OLLAMA, the system processes student answers, cross-references them with a custom historical knowledge base using Retrieval-Augmented Generation (RAG), and validates facts against a domain-specific ontology. 

The core grading engine is powered by an **Agentic Architecture** that separates tasks (retrieval, ontology mapping, grading, and justification) to provide an explainable score out of 20 marks.

## ✨ Key Features
* **100% Offline Operation:** No internet connectivity is required during inference, ensuring data privacy and compliance with assignment constraints.
* **Agent-Based Architecture:** Specialized agents handle distinct responsibilities to improve reliability and reduce hallucinations.
* **Sinhala Language Processing:** Capable of comprehending and evaluating complex historical text in Sinhala.
* **RAG & Ontology Integration:** Retrieves relevant historical evidence and validates relationships/entities using a structured knowledge base.
* **Explainable Scoring:** Provides a transparent mark breakdown per grading criteria, grounded in retrieved evidence.
* **Interactive UI:** A clean, user-friendly Streamlit interface for selecting questions, inputting answers, and viewing the detailed score breakdown.

## 🏗️ System Architecture
1. **Retrieval Agent:** Queries the local vector store for historical context relevant to the student's answer.
2. **Ontology Agent:** Validates historical entities, timelines, and relationships mentioned in the text.
3. **Grading Agent:** Compares the validated text against a strict 20-mark grading rubric.
4. **Justification Agent:** Generates an explainable summary detailing exactly why marks were awarded or deducted.

## ⚙️ Prerequisites
To run this project locally, you will need:
* **Python 3.8+**
* **OLLAMA** installed and running on your local machine.
* A compatible open-source LLM pulled via OLLAMA (e.g., `llama3`, `mistral`, or a custom fine-tune).

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/sinhala-answer-scorer.git](https://github.com/yourusername/sinhala-answer-scorer.git)
   cd sinhala-answer-scorer