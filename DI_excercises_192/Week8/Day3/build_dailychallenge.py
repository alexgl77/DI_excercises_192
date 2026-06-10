"""
Generate dailychallenge.ipynb for Week 8 Day 3:
RAG with LangChain on databricks/databricks-dolly-15k — load + chunk + embed +
FAISS + Intel/dynamic_tinybert extractive QA + flan-t5-small generative QA via
RetrievalQA.
"""

import json
import os


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code(*lines):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [l + "\n" for l in lines],
    }


cells = [
    md(
        "# Daily Challenge — Build a RAG System (LangChain + Hugging Face)",
        "",
        "**Course:** Developers Institute  **Week 8 - Day 3**  ",
        "**Author:** Alex Goldbaum",
        "",
        "End-to-end RAG over the **`databricks/databricks-dolly-15k`** dataset:",
        "",
        "1. Install + import dependencies.",
        "2. Load the dataset with `HuggingFaceDatasetLoader`.",
        "3. Split documents with `RecursiveCharacterTextSplitter`.",
        "4. Embed with `sentence-transformers/all-MiniLM-L6-v2`.",
        "5. Build a **FAISS** vector store + retriever.",
        "6. Run **two** answering strategies on the same retrieved chunks:",
        "   - the **extractive QA** model `Intel/dynamic_tinybert` (the model the",
        "     brief specifies) called directly on the retrieved context.",
        "   - the **generative** `RetrievalQA` chain with `chain_type='refine'`.",
        "7. Test with the question *\"What is cheesemaking?\"* (plus a few extras).",
        "",
        "**Note about the model choice.** `Intel/dynamic_tinybert` is an *extractive*",
        "span-prediction QA model — it returns a substring of a single context, not a",
        "free-form answer. LangChain's `RetrievalQA(chain_type='refine')` expects a",
        "*generative* LM, so we keep `Intel/dynamic_tinybert` for the direct",
        "extractive route (closest to the brief) and use `google/flan-t5-small` for",
        "the LangChain chain. The retriever + FAISS + chunking are identical in both.",
    ),

    # ============================================================
    # 1. Setup
    # ============================================================
    md("## 1. Set Up Your Environment"),
    code(
        "!pip install -q langchain langchain-community langchain-huggingface",
        "!pip install -q torch transformers sentence-transformers",
        "!pip install -Uq datasets",
        "!pip install -q faiss-cpu",
    ),
    code(
        "import warnings",
        "warnings.filterwarnings('ignore')",
        "",
        "import torch",
        "device = 'cuda' if torch.cuda.is_available() else 'cpu'",
        "print('Device:', device)",
    ),

    # ============================================================
    # 2. Load dataset
    # ============================================================
    md("## 2. Load the Dataset"),
    code(
        "from langchain_community.document_loaders import HuggingFaceDatasetLoader",
        "",
        "dataset_name = 'databricks/databricks-dolly-15k'",
        "page_content_column = 'context'",
        "",
        "loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)",
        "data = loader.load()",
        "",
        "print(f'Loaded {len(data)} documents.')",
        "print('First 2 entries:')",
        "for d in data[:2]:",
        "    print(' ----')",
        "    print(' page_content[:200]:', d.page_content[:200])",
        "    print(' metadata         :', d.metadata)",
    ),
    md(
        "**A real-world wrinkle.** Many Dolly examples have an *empty* `context`",
        "field because the question can be answered without retrieved knowledge.",
        "Those rows produce empty `page_content` and just dilute the index — we",
        "drop them before chunking.",
    ),
    code(
        "data = [d for d in data if d.page_content and d.page_content.strip()]",
        "print(f'After dropping empty contexts: {len(data)} documents.')",
    ),

    # ============================================================
    # 3. Split
    # ============================================================
    md("## 3. Split the Documents into Chunks"),
    code(
        "from langchain.text_splitter import RecursiveCharacterTextSplitter",
        "",
        "text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)",
        "docs = text_splitter.split_documents(data)",
        "",
        "print(f'Split into {len(docs)} chunks.')",
        "print('First chunk:')",
        "print(' page_content[:300]:', docs[0].page_content[:300])",
        "print(' metadata         :', docs[0].metadata)",
    ),

    # ============================================================
    # 4. Embed
    # ============================================================
    md("## 4. Embed the Text"),
    code(
        "from langchain_huggingface import HuggingFaceEmbeddings",
        "",
        "modelPath = 'sentence-transformers/all-MiniLM-l6-v2'",
        "model_kwargs = {'device': device}",
        "encode_kwargs = {'normalize_embeddings': False}",
        "",
        "embeddings = HuggingFaceEmbeddings(",
        "    model_name=modelPath,",
        "    model_kwargs=model_kwargs,",
        "    encode_kwargs=encode_kwargs,",
        ")",
        "",
        "# Optional sanity check on the embeddings",
        "text = 'This is a test document.'",
        "query_result = embeddings.embed_query(text)",
        "print('Embedding dim:', len(query_result))",
        "print('First 3 values:', query_result[:3])",
    ),

    # ============================================================
    # 5. FAISS
    # ============================================================
    md("## 5. Create a FAISS Vector Store"),
    code(
        "from langchain_community.vectorstores import FAISS",
        "",
        "# This step takes a few minutes — it embeds every chunk and indexes it.",
        "db = FAISS.from_documents(docs, embeddings)",
        "print(f'FAISS index built with {db.index.ntotal} vectors.')",
    ),

    # ============================================================
    # 6. LLM + extractive QA + generative QA
    # ============================================================
    md(
        "## 6. Prepare the LLMs (Extractive and Generative)",
        "",
        "### 6.1 The extractive model from the brief — `Intel/dynamic_tinybert`",
        "",
        "It returns the **span** in the supplied context that most likely answers the",
        "question. It is fast and accurate when the answer is literally written in",
        "the context.",
    ),
    code(
        "from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline",
        "",
        "qa_model_name = 'Intel/dynamic_tinybert'",
        "qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name, padding=True,",
        "                                            truncation=True, max_length=512)",
        "qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)",
        "",
        "question_answerer = pipeline(",
        "    'question-answering',",
        "    model=qa_model,",
        "    tokenizer=qa_tokenizer,",
        "    device=0 if torch.cuda.is_available() else -1,",
        ")",
        "print('Extractive QA pipeline ready.')",
    ),
    md(
        "### 6.2 The generative model used inside the LangChain `RetrievalQA` chain",
        "",
        "`google/flan-t5-small` is a tiny instruction-tuned seq2seq model that runs",
        "anywhere and integrates cleanly with `HuggingFacePipeline` and the",
        "`refine` chain type.",
    ),
    code(
        "from transformers import AutoModelForSeq2SeqLM",
        "from langchain_huggingface import HuggingFacePipeline",
        "",
        "gen_model_name = 'google/flan-t5-small'",
        "gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)",
        "gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)",
        "",
        "gen_pipeline = pipeline(",
        "    'text2text-generation',",
        "    model=gen_model,",
        "    tokenizer=gen_tokenizer,",
        "    max_new_tokens=256,",
        "    device=0 if torch.cuda.is_available() else -1,",
        ")",
        "",
        "llm = HuggingFacePipeline(",
        "    pipeline=gen_pipeline,",
        "    model_kwargs={'temperature': 0.7, 'max_length': 512},",
        ")",
        "print('Generative LLM ready.')",
    ),

    # ============================================================
    # 7. RetrievalQA chain
    # ============================================================
    md("## 7. Build the Retrieval QA Chain"),
    code(
        "from langchain.chains import RetrievalQA",
        "",
        "retriever = db.as_retriever(search_kwargs={'k': 4})",
        "",
        "qa = RetrievalQA.from_chain_type(",
        "    llm=llm,",
        "    chain_type='refine',",
        "    retriever=retriever,",
        "    return_source_documents=False,",
        ")",
        "print('RetrievalQA chain ready (chain_type=\"refine\", k=4).')",
    ),

    # ============================================================
    # 8. Test
    # ============================================================
    md(
        "## 8. Test the RAG System",
        "",
        "### 8.1 The chain output — generative answer",
    ),
    code(
        "question = 'What is cheesemaking?'",
        "",
        "result = qa.invoke({'query': question})",
        "answer = result['result'] if isinstance(result, dict) else result",
        "print(f'Q: {question}')",
        "print(f'A (chain): {answer}')",
    ),
    md(
        "### 8.2 The extractive answer (`Intel/dynamic_tinybert` on the retrieved chunks)",
        "",
        "We retrieve the top-4 chunks ourselves, concatenate them, and ask the",
        "extractive model to find the answer span. This is the natural way to use",
        "`Intel/dynamic_tinybert` in a RAG setup.",
    ),
    code(
        "def extractive_rag_answer(question: str, k: int = 4):",
        "    hits = db.similarity_search(question, k=k)",
        "    # Run the QA model against every retrieved chunk, keep the highest-scoring span",
        "    candidates = []",
        "    for h in hits:",
        "        out = question_answerer(question=question, context=h.page_content)",
        "        candidates.append({",
        "            'answer': out['answer'],",
        "            'score': float(out['score']),",
        "            'source': h.metadata,",
        "            'context_preview': h.page_content[:200],",
        "        })",
        "    candidates.sort(key=lambda c: c['score'], reverse=True)",
        "    return candidates[0], candidates",
        "",
        "",
        "best, all_candidates = extractive_rag_answer(question, k=4)",
        "print(f'Q: {question}')",
        "print(f'A (extractive): {best[\"answer\"]}  (confidence {best[\"score\"]:.3f})')",
        "print('Top candidate context preview:')",
        "print(best['context_preview'])",
    ),
    md("### 8.3 Run several more questions through both pipelines"),
    code(
        "questions = [",
        "    'What is cheesemaking?',",
        "    'What is photosynthesis?',",
        "    'Who developed the theory of relativity?',",
        "    'What does an octopus eat?',",
        "    'What is the Great Wall of China?',",
        "]",
        "",
        "for q in questions:",
        "    print(f'>>> {q}')",
        "    # Generative",
        "    try:",
        "        gen = qa.invoke({'query': q})",
        "        gen_answer = gen['result'] if isinstance(gen, dict) else gen",
        "    except Exception as e:",
        "        gen_answer = f'(chain error: {e})'",
        "    print(f'   chain      : {gen_answer}')",
        "",
        "    # Extractive",
        "    try:",
        "        best, _ = extractive_rag_answer(q, k=4)",
        "        print(f'   extractive : {best[\"answer\"]}  (conf={best[\"score\"]:.3f})')",
        "    except Exception as e:",
        "        print(f'   extractive : (error: {e})')",
        "    print()",
    ),

    md(
        "## Summary",
        "",
        "- We built a fully local RAG system over **`databricks/databricks-dolly-",
        "  15k`** with LangChain orchestrating the pieces and Hugging Face providing",
        "  every model.",
        "- Documents loaded → chunked → embedded with `all-MiniLM-L6-v2` → indexed in",
        "  **FAISS** → retrieved by similarity.",
        "- We then exposed **two answering modes** over the same retrieved chunks:",
        "    - **Generative** via `RetrievalQA(chain_type='refine')` with",
        "      `flan-t5-small` — natural-language answers, robust to missing spans.",
        "    - **Extractive** via the `Intel/dynamic_tinybert` QA pipeline applied to",
        "      each chunk — high-confidence answers when the text contains them.",
        "- For a question whose answer is literally written in the dataset,",
        "  extractive often gives the cleanest output; for paraphrased or general",
        "  questions, the generative chain wins. A production RAG system commonly",
        "  layers both — extractive QA as a sanity check on top of a generative LM.",
        "",
        "**Tuning levers exposed.**",
        "- `chunk_size` / `chunk_overlap` in the text splitter.",
        "- `k` in the retriever.",
        "- The generator model (`flan-t5-small` → `flan-t5-base` →",
        "  `mistralai/Mistral-7B-Instruct` and beyond).",
        "- The chain type (`stuff`, `map_reduce`, `refine`, `map_rerank`).",
    ),
]


nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out_path = os.path.join(os.path.dirname(__file__), "dailychallenge.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook generated: {out_path}")
print(f"Size: {os.path.getsize(out_path)} bytes")
print(f"Cells: {len(cells)}")
