"""
Generate dailychallenge.ipynb for Week 8 Day 3:
Build a local Retrieval-Augmented Generation system with LangChain,
FAISS, sentence-transformers, and the small instruction-tuned model
google/flan-t5-small. No API keys required.
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
        "# XP Exercises — RAG with LangChain (Local, No API Keys)",
        "",
        "**Course:** Developers Institute  **Week 8 - Day 3**  ",
        "**Author:** Alex Goldbaum",
        "",
        "Build a tiny but **fully local** RAG pipeline:",
        "",
        "1. Setup + imports.",
        "2. Load the `m-ric/huggingface_doc` dataset (`train[:200]`).",
        "3. Convert rows to LangChain `Document` objects.",
        "4. Chunk with `RecursiveCharacterTextSplitter` — try two parameter sets.",
        "5. Embed chunks with `all-MiniLM-L6-v2`, store in **FAISS**, build a",
        "   retriever (`k=2`, `k=4`, `k=6`).",
        "6. Retrieval sanity check before generating any answer.",
        "7. Plug a local **`flan-t5-small`** into a `RetrievalQA` chain and ask",
        "   real questions — printing the answer **and** the source chunks so we",
        "   can audit the model.",
    ),

    # ============================================================
    # 1. Setup
    # ============================================================
    md("## 1. Setup and Imports"),
    code(
        "!pip install -q datasets langchain langchain-community langchain-huggingface sentence-transformers faiss-cpu transformers",
    ),
    code(
        "import warnings",
        "warnings.filterwarnings('ignore')",
        "",
        "import os",
        "import torch",
        "",
        "from datasets import load_dataset",
        "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline",
        "",
        "# LangChain components",
        "from langchain.schema import Document",
        "from langchain.text_splitter import RecursiveCharacterTextSplitter",
        "from langchain_community.vectorstores import FAISS",
        "from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline",
        "from langchain.chains import RetrievalQA",
        "",
        "device = 0 if torch.cuda.is_available() else -1",
        "print('CUDA available:', torch.cuda.is_available())",
    ),

    # ============================================================
    # 2. Load dataset
    # ============================================================
    md("## 2. Load the Dataset"),
    code(
        "raw = load_dataset('m-ric/huggingface_doc', split='train[:200]')",
        "print('Columns:', raw.column_names)",
        "print('Number of rows:', len(raw))",
        "print()",
        "print('--- One example row ---')",
        "example = raw[0]",
        "for k, v in example.items():",
        "    preview = (v[:300] + '...') if isinstance(v, str) and len(v) > 300 else v",
        "    print(f'{k}: {preview}')",
    ),

    # ============================================================
    # 3. Convert to LangChain Documents
    # ============================================================
    md(
        "## 3. Convert Rows into LangChain `Document` Objects",
        "",
        "Each row becomes a `Document` with the raw `text` in `page_content` and the",
        "`source` URL kept inside `metadata` — that way every retrieved chunk can be",
        "traced back to its file in the audit step at the end.",
    ),
    code(
        "documents = [",
        "    Document(",
        "        page_content=row['text'],",
        "        metadata={'source': row.get('source', f'doc_{i}')},",
        "    )",
        "    for i, row in enumerate(raw)",
        "]",
        "print(f'Built {len(documents)} Documents.')",
        "print('First document metadata:', documents[0].metadata)",
        "print('First 200 chars of page_content:', documents[0].page_content[:200])",
    ),

    # ============================================================
    # 4. Chunking
    # ============================================================
    md(
        "## 4. Chunk the Documents",
        "",
        "Embedding models have a token limit (`all-MiniLM-L6-v2` truncates at 256",
        "tokens), and large documents would otherwise overwhelm a single embedding",
        "vector — diluting the signal of any individual paragraph. We use",
        "`RecursiveCharacterTextSplitter`, which respects natural break points",
        "(paragraphs → sentences → words) when shrinking text.",
        "",
        "We compare **two configurations** to see the effect on chunk count and",
        "content boundaries.",
    ),
    code(
        "# Configuration A — bigger chunks with light overlap",
        "splitter_A = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)",
        "chunks_A = splitter_A.split_documents(documents)",
        "",
        "# Configuration B — smaller chunks with more overlap",
        "splitter_B = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)",
        "chunks_B = splitter_B.split_documents(documents)",
        "",
        "print(f'Config A (1000/100): {len(chunks_A)} chunks')",
        "print(f'Config B  (400/80) : {len(chunks_B)} chunks')",
        "",
        "print('\\nExample chunk from config A (first 500 chars):')",
        "print(chunks_A[0].page_content[:500])",
        "print()",
        "print('Example chunk from config B (first 500 chars):')",
        "print(chunks_B[0].page_content[:500])",
    ),
    md(
        "**Observation.** Config B produces ~2-3× more chunks. Smaller chunks give",
        "**more precise retrieval** (the relevant paragraph is a higher fraction of",
        "the retrieved text) but lose some surrounding context. Bigger chunks keep",
        "the context but dilute the embedding when the query only touches part of",
        "the chunk. The right answer depends on the typical question length and the",
        "downstream model's context window.",
    ),

    # ============================================================
    # 5. Vector store + retriever
    # ============================================================
    md(
        "## 5. Build the Vector Store and Retriever",
        "",
        "Embed every chunk with `sentence-transformers/all-MiniLM-L6-v2` (384",
        "dimensions), store the vectors in **FAISS**, and expose a retriever. We",
        "use Config B chunks because they give crisper paragraph-level retrieval —",
        "and we compare `k=2`, `k=4`, `k=6` on the same query.",
    ),
    code(
        "embeddings = HuggingFaceEmbeddings(",
        "    model_name='sentence-transformers/all-MiniLM-L6-v2',",
        "    model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},",
        ")",
        "",
        "vector_store = FAISS.from_documents(chunks_B, embeddings)",
        "print(f'FAISS index built with {vector_store.index.ntotal} vectors.')",
    ),
    code(
        "retriever_k2 = vector_store.as_retriever(search_kwargs={'k': 2})",
        "retriever_k4 = vector_store.as_retriever(search_kwargs={'k': 4})",
        "retriever_k6 = vector_store.as_retriever(search_kwargs={'k': 6})",
        "",
        "probe_query = 'What is the Hugging Face Transformers library?'",
        "for name, r in [('k=2', retriever_k2), ('k=4', retriever_k4), ('k=6', retriever_k6)]:",
        "    hits = r.invoke(probe_query)",
        "    print(f'-- {name}: returned {len(hits)} chunks --')",
        "    for i, h in enumerate(hits):",
        "        print(f'  [{i+1}] {h.metadata.get(\"source\")[:80]}  '",
        "              f'| {h.page_content[:120]!r}...')",
        "    print()",
    ),

    # ============================================================
    # 6. Retrieval sanity check
    # ============================================================
    md(
        "## 6. Retrieval Sanity Check",
        "",
        "Before plugging the LM in, we read the retrieved chunks carefully. If the",
        "chunks look irrelevant we re-tune `chunk_size`, `chunk_overlap`, or `k` —",
        "**never trust an LLM's answer when the retrieval upstream is bad**.",
    ),
    code(
        "sanity_query = 'How do I fine-tune a transformer with the Hugging Face Trainer?'",
        "results = retriever_k4.invoke(sanity_query)",
        "",
        "print(f'Query: {sanity_query!r}')",
        "print(f'Got {len(results)} chunks.')",
        "for i, r in enumerate(results, 1):",
        "    print(f'\\n--- Chunk {i} (source: {r.metadata.get(\"source\", \"?\")}) ---')",
        "    print(r.page_content[:600])",
    ),
    md(
        "If those chunks discuss `Trainer`, `TrainingArguments`, fine-tuning, etc.,",
        "the retrieval is working. If they look off-topic, this is the moment to",
        "rebuild the index with smaller chunks or a different `k`.",
    ),

    # ============================================================
    # 7. RAG question answering
    # ============================================================
    md(
        "## 7. RAG Question Answering with `flan-t5-small`",
        "",
        "`flan-t5-small` is an instruction-tuned seq2seq model that fits in a few",
        "hundred MB of RAM — perfect for a local RAG demo. We wrap it in a",
        "Hugging Face `pipeline`, then in LangChain's `HuggingFacePipeline`, then",
        "plug it into a `RetrievalQA` chain. The chain handles prompt assembly,",
        "retrieval, and answer generation in one call.",
    ),
    code(
        "MODEL_ID = 'google/flan-t5-small'",
        "",
        "tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)",
        "lm_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)",
        "",
        "gen_pipeline = pipeline(",
        "    'text2text-generation',",
        "    model=lm_model,",
        "    tokenizer=tokenizer,",
        "    max_new_tokens=256,",
        "    device=device,",
        ")",
        "",
        "llm = HuggingFacePipeline(pipeline=gen_pipeline)",
    ),
    code(
        "qa_chain = RetrievalQA.from_chain_type(",
        "    llm=llm,",
        "    retriever=retriever_k4,",
        "    chain_type='stuff',",
        "    return_source_documents=True,",
        ")",
    ),
    code(
        "questions = [",
        "    'What is the Transformers library used for?',",
        "    'How do I install Hugging Face datasets?',",
        "    'What is the Hugging Face Hub?',",
        "    'How can I fine-tune a model with Trainer?',",
        "    'What are pipelines in transformers?',",
        "]",
        "",
        "for q in questions:",
        "    out = qa_chain.invoke({'query': q})",
        "    print(f'>>> {q}')",
        "    print(f'ANSWER : {out[\"result\"].strip()}')",
        "    print('SOURCES:')",
        "    seen = set()",
        "    for doc in out['source_documents']:",
        "        src = doc.metadata.get('source', '?')",
        "        if src not in seen:",
        "            print(f'   - {src}')",
        "            seen.add(src)",
        "    print()",
    ),
    md(
        "**Why the sources are printed.** A core RAG hygiene rule: every answer",
        "must be auditable back to the chunks that produced it. If the answer",
        "looks wrong, the sources tell us **why** — was it a retrieval problem",
        "(wrong chunks) or a generation problem (model ignored relevant chunks)?",
    ),

    md(
        "## Summary",
        "",
        "- We built a **fully local** RAG pipeline — no API keys, no paid services.",
        "- The pipeline reads from `m-ric/huggingface_doc`, chunks the docs with",
        "  `RecursiveCharacterTextSplitter`, embeds them with",
        "  `all-MiniLM-L6-v2`, stores them in a **FAISS** vector store and exposes",
        "  a **retriever** controlled by `k`.",
        "- The generator is **`flan-t5-small`**, wired via LangChain's",
        "  `HuggingFacePipeline` into a `RetrievalQA` chain.",
        "- Every answer is returned together with its **source documents**, which",
        "  is the only way to keep a RAG system trustworthy in practice.",
        "",
        "**Tuning levers we exposed.**",
        "- `chunk_size` / `chunk_overlap` (Config A vs Config B) — precision vs context.",
        "- `k` in the retriever (`2`, `4`, `6`) — coverage vs noise.",
        "- The generator itself — swap `flan-t5-small` for a larger model",
        "  (`flan-t5-base`, `Llama-3.1-8B-Instruct`, GPT-4) and the same retrieval",
        "  pipeline produces dramatically better answers.",
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

out_path = os.path.join(os.path.dirname(__file__), "exercisesxp.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook generated: {out_path}")
print(f"Size: {os.path.getsize(out_path)} bytes")
print(f"Cells: {len(cells)}")
