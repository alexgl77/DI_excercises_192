"""
Generate exercisesxp.ipynb for Week 8 Day 2:
End-to-end RAG pipeline — FAISS + ChromaDB + Hugging Face LLM
on the labelled_newscatcher_dataset (or AG News as a fallback).
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
        "# XP Exercises — RAG Pipeline (FAISS + ChromaDB + Hugging Face)",
        "",
        "**Course:** Developers Institute  **Week 8 - Day 2**  ",
        "**Author:** Alex Goldbaum",
        "",
        "Build a full **Retrieval-Augmented Generation** pipeline in five exercises:",
        "",
        "1. Load and inspect the news dataset.",
        "2. Vectorize titles with `sentence-transformers` (`all-MiniLM-L6-v2`).",
        "3. Index the embeddings in **FAISS** and run cosine similarity search.",
        "4. Mirror the same workflow in **ChromaDB** (which embeds for us).",
        "5. Generate context-aware answers with a **Hugging Face** causal LM.",
        "",
        "**Dataset note.** The brief uses `labelled_newscatcher_dataset.csv`. If you",
        "have that file locally, drop it next to the notebook and it will be picked",
        "up. Otherwise we fall back to `ag_news` (same shape: `title` + `topic`) so",
        "the rest of the pipeline runs unchanged.",
    ),

    md("## Exercise 1 — Data Loading and Preparation"),
    md("### 1. Install + import the required libraries"),
    code(
        "!pip install -q 'faiss-cpu==1.7.4' 'chromadb' 'numpy<2' sentence-transformers transformers datasets",
    ),
    code(
        "import os, json, warnings",
        "warnings.filterwarnings('ignore')",
        "",
        "import numpy as np",
        "import pandas as pd",
        "import faiss",
        "",
        "from sentence_transformers import SentenceTransformer, InputExample",
        "import chromadb",
        "from chromadb.config import Settings",
        "from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline",
        "",
        "import torch",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
        "print('Device:', device)",
        "",
        "# Cache directory the brief asks for",
        "os.makedirs('cache', exist_ok=True)",
    ),
    md("### 2. Load the dataset into a pandas DataFrame"),
    code(
        "# Try the original CSV first, fall back to ag_news from Hugging Face datasets.",
        "path = 'labelled_newscatcher_dataset.csv'",
        "",
        "if os.path.exists(path):",
        "    pdf = pd.read_csv(path, sep=';')  # the original CSV uses semicolons",
        "    print(f'Loaded {len(pdf)} rows from {path}')",
        "else:",
        "    print(f'{path} not found - falling back to ag_news from Hugging Face.')",
        "    from datasets import load_dataset",
        "    ag = load_dataset('ag_news', split='train[:5000]')",
        "    ag_label_names = ag.features['label'].names",
        "    pdf = pd.DataFrame({",
        "        'topic': [ag_label_names[i] for i in ag['label']],",
        "        'title': ag['text'],",
        "    })",
        "    print(f'Loaded {len(pdf)} rows from ag_news ({ag_label_names}).')",
    ),
    md("### 3. Add a unique identifier column"),
    code(
        "pdf['id'] = pdf.index.astype(str)",
        "print(pdf.columns.tolist())",
    ),
    md("### 4. Inspect the data"),
    code(
        "print('Shape:', pdf.shape)",
        "print('Missing values per column:')",
        "print(pdf.isna().sum())",
        "print('\\nTopic distribution:')",
        "print(pdf['topic'].value_counts().head())",
        "pdf.head()",
    ),
    md("### 5. Take a subset for faster iteration"),
    code(
        "pdf_subset = pdf.head(1000).reset_index(drop=True)",
        "print(f'Subset shape: {pdf_subset.shape}')",
    ),

    # ============================================================
    # Exercise 2 — Vectorization
    # ============================================================
    md("## Exercise 2 — Vectorization with Sentence Transformers"),
    md("### 3. Helper that formats one title as an `InputExample`"),
    code(
        "def example_create_fn(doc1: pd.Series) -> InputExample:",
        "    \"\"\"Wrap a single title into the InputExample structure used by sentence-transformers.\"\"\"",
        "    return InputExample(texts=[doc1])",
        "",
        "",
        "# Apply across the subset",
        "faiss_train_examples = pdf_subset.apply(",
        "    lambda x: example_create_fn(x['title']), axis=1,",
        ").tolist()",
        "faiss_train_examples[:10]",
    ),
    md("### 5. Initialize the embedding model"),
    code(
        "model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')",
        "print('Embedding model loaded.')",
    ),
    md("### 6. Extract titles + 7. Generate embeddings"),
    code(
        "titles_list = pdf_subset['title'].tolist()",
        "",
        "faiss_title_embedding = model.encode(",
        "    titles_list,",
        "    show_progress_bar=True,",
        "    convert_to_numpy=True,",
        ")",
        "",
        "# 8. Confirm shape — should be (n_titles, 384)",
        "print('Number of embeddings :', len(faiss_title_embedding))",
        "print('Embedding dimension  :', len(faiss_title_embedding[0]))",
        "print('NumPy shape          :', faiss_title_embedding.shape)",
    ),

    # ============================================================
    # Exercise 3 — FAISS
    # ============================================================
    md("## Exercise 3 — FAISS Indexing and Search"),
    code(
        "pdf_to_index = pdf_subset.copy()",
        "id_index = np.array(pdf_to_index['id'].values, dtype=np.int64)",
        "",
        "# 3. Normalize the embeddings (so inner product == cosine similarity)",
        "content_encoded_normalized = faiss_title_embedding.copy().astype('float32')",
        "faiss.normalize_L2(content_encoded_normalized)",
        "",
        "# 4. Build the index",
        "index_content = faiss.IndexIDMap(faiss.IndexFlatIP(len(faiss_title_embedding[0])))",
        "index_content.add_with_ids(content_encoded_normalized, id_index)",
        "print(f'FAISS index built with {index_content.ntotal} vectors of dim',",
        "      faiss_title_embedding.shape[1])",
    ),
    md("### 5. Search function"),
    code(
        "def search_content(query, pdf_to_index, k=3):",
        "    # Encode the query and normalize",
        "    query_vector = model.encode([query]).astype('float32')",
        "    faiss.normalize_L2(query_vector)",
        "",
        "    # Search the index",
        "    top_k = index_content.search(query_vector, k)",
        "    ids = top_k[1][0].tolist()",
        "    similarities = top_k[0][0].tolist()",
        "",
        "    # Retrieve the matching rows",
        "    results = pdf_to_index.loc[ids].copy().reset_index(drop=True)",
        "    results['similarities'] = similarities",
        "    return results",
    ),
    md("### 6. Test the search"),
    code(
        "search_content('animal', pdf_to_index, k=5)",
    ),

    # ============================================================
    # Exercise 4 — ChromaDB
    # ============================================================
    md("## Exercise 4 — ChromaDB Collection and Querying"),
    md("### 2. Initialize a ChromaDB client and (re)create a collection"),
    code(
        "chroma_client = chromadb.Client()",
        "collection_name = 'my_news'",
        "",
        "existing = [c.name for c in chroma_client.list_collections()]",
        "if collection_name in existing:",
        "    chroma_client.delete_collection(name=collection_name)",
        "",
        "print(f\"Creating collection: '{collection_name}'\")",
        "collection = chroma_client.create_collection(name=collection_name)",
    ),
    md("### 3. Add the first 100 titles + metadata + IDs"),
    code(
        "collection.add(",
        "    documents=pdf_subset['title'][:100].tolist(),",
        "    metadatas=[{'topic': topic} for topic in pdf_subset['topic'][:100].tolist()],",
        "    ids=[str(i) for i in pdf_subset['id'][:100].tolist()],",
        ")",
        "print(f'Collection now has {collection.count()} documents.')",
    ),
    md("### 4. Query the collection"),
    code(
        "results = collection.query(",
        "    query_texts=['space'],",
        "    n_results=10,",
        ")",
        "",
        "print(json.dumps(results, indent=2, default=str)[:2000])",
    ),

    # ============================================================
    # Exercise 5 — QA with Hugging Face
    # ============================================================
    md("## Exercise 5 — Question Answering with a Hugging Face LLM"),
    md(
        "### 2. Load model + tokenizer",
        "",
        "We use **GPT-2** as the smallest, fastest, freely-available causal LM that",
        "fits in any Colab runtime — perfect for a teaching example. For real RAG",
        "deployments you would swap it for a larger / instruction-tuned model",
        "(`google/flan-t5-base`, `mistralai/Mistral-7B-Instruct`, etc.).",
    ),
    code(
        "model_id = 'gpt2'",
        "tokenizer = AutoTokenizer.from_pretrained(model_id)",
        "if tokenizer.pad_token is None:",
        "    tokenizer.pad_token = tokenizer.eos_token",
        "lm_model = AutoModelForCausalLM.from_pretrained(model_id)",
    ),
    md("### 3. Text generation pipeline"),
    code(
        "pipe = pipeline(",
        "    'text-generation',",
        "    model=lm_model,",
        "    tokenizer=tokenizer,",
        "    max_new_tokens=120,",
        "    device_map='auto',",
        "    pad_token_id=tokenizer.eos_token_id,",
        ")",
    ),
    md("### 4. Build the prompt template + 5. Generate the answer"),
    code(
        "question = \"What's the latest news on space development?\"",
        "context = ' '.join([f'#{str(i)}' for i in results['documents'][0]])",
        "prompt_template = f\"Relevant context: {context}\\n\\n The user's question: {question}\"",
        "",
        "print('Prompt (first 600 chars):')",
        "print(prompt_template[:600])",
        "print()",
        "",
        "lm_response = pipe(prompt_template)",
        "print('=== Generated answer ===')",
        "print(lm_response[0]['generated_text'])",
    ),
    md(
        "### 6. Experimenting with different prompts and context sizes",
        "",
        "Below we vary both the question and the number of retrieved documents to",
        "see how the answer changes. With more context the model has more evidence",
        "to draw from; with less, it leans more heavily on its pretraining.",
    ),
    code(
        "def rag_answer(question: str, n_results: int = 5):",
        "    res = collection.query(query_texts=[question], n_results=n_results)",
        "    context = ' '.join([f'#{d}' for d in res['documents'][0]])",
        "    prompt = f\"Relevant context: {context}\\n\\nThe user's question: {question}\"",
        "    return pipe(prompt)[0]['generated_text']",
        "",
        "",
        "for q in [",
        "    'What does this news say about technology?',",
        "    'Summarize the most important sports news.',",
        "    'Which businesses are mentioned?',",
        "]:",
        "    print(f'>>> {q}')",
        "    print(rag_answer(q, n_results=3))",
        "    print()",
    ),

    md(
        "## Summary",
        "",
        "- We loaded the news dataset, gave it a stable identifier column and",
        "  trimmed it to 1,000 rows for fast iteration.",
        "- We embedded the titles with `all-MiniLM-L6-v2` (384-dim sentence",
        "  embeddings) and indexed them in **FAISS** with `IndexFlatIP +",
        "  IndexIDMap` plus L2 normalization, which gives us cosine similarity",
        "  search in a few lines.",
        "- We mirrored the same retrieval workflow in **ChromaDB**, which embeds",
        "  documents and queries for us — handier for higher-level apps and",
        "  metadata-aware filtering.",
        "- Finally we wired the retrieved context into a **GPT-2** causal LM and",
        "  generated answers from a prompt that mixes question + retrieved context",
        "  — the textbook RAG pattern.",
        "",
        "**Practical takeaways.**",
        "- FAISS is the library to reach for when you control the embeddings and",
        "  want maximum throughput; ChromaDB is the database when you want a higher-",
        "  level API with metadata and persistence.",
        "- The generator's quality dominates the end-user experience: a strong",
        "  retriever paired with a weak generator (like GPT-2) still produces poor",
        "  answers. In production, swap GPT-2 for a larger instruction-tuned model",
        "  and you will see the same retrieved context produce much better outputs.",
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
