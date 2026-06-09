"""
Generate dailychallenge.ipynb for Week 8 Day 2:
Pinecone Serverless Reranking — sample reranker run + serverless index
for medical notes + semantic search + reranking with bge-reranker-v2-m3.
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
        "# Daily Challenge — Pinecone Serverless Reranking in Action",
        "",
        "**Course:** Developers Institute  **Week 8 - Day 2**  ",
        "**Author:** Alex Goldbaum",
        "",
        "End-to-end Pinecone workflow: run the **`bge-reranker-v2-m3`** reranker on",
        "a tiny apple-fruit / Apple-company test set, then build a real",
        "**serverless index** of sample medical notes, retrieve by semantic search",
        "and rerank for clinical relevance.",
        "",
        "**⚠️ You need a Pinecone account.** Sign up at https://pinecone.io and grab",
        "your API key (Dashboard → API Keys). The notebook will prompt for it if",
        "it is not already set in the environment.",
    ),

    # ============================================================
    # Part 1: Reranker model demo
    # ============================================================
    md("## Part 1 — Load Documents & Execute the Reranking Model"),
    code(
        "!pip install -q -U pinecone==6.0.1 pinecone-notebooks",
    ),
    md("### 2. Authenticate with Pinecone"),
    code(
        "import os",
        "",
        "if not os.environ.get('PINECONE_API_KEY'):",
        "    try:",
        "        from pinecone_notebooks.colab import Authenticate",
        "        Authenticate()",
        "    except Exception:",
        "        # Local fallback — paste the key when prompted",
        "        import getpass",
        "        os.environ['PINECONE_API_KEY'] = getpass.getpass('Pinecone API key: ')",
        "",
        "assert os.environ.get('PINECONE_API_KEY'), 'PINECONE_API_KEY is required.'",
        "print('Pinecone API key configured.')",
    ),
    md("### 3. Instantiate the Pinecone client"),
    code(
        "from pinecone import Pinecone",
        "",
        "api_key = os.environ.get('PINECONE_API_KEY')",
        "pc = Pinecone(api_key=api_key)",
        "print('Pinecone client ready.')",
    ),
    md(
        "### 4. Define a query + 5 test documents",
        "",
        "Mix references to **Apple** (the company) and **apple** (the fruit) so we",
        "can see whether the reranker disambiguates the meaning from context.",
    ),
    code(
        "query = \"Tell me about Apple's products\"",
        "",
        "documents = [",
        "    'An apple is a sweet, crisp fruit that grows on apple trees and is widely cultivated worldwide.',",
        "    'Apple Inc. designs and sells the iPhone, iPad, Mac computers and AirPods, plus services like iCloud and Apple Music.',",
        "    'Honeycrisp and Granny Smith are two popular apple cultivars known for their distinctive flavor profiles.',",
        "    'Apple released the new MacBook Pro with the M3 chip, offering huge performance gains for creative professionals.',",
        "    'A medium apple contains around 95 calories and is an excellent source of dietary fiber and vitamin C.',",
        "]",
    ),
    md("### 5. Call the reranker"),
    code(
        "reranked = pc.inference.rerank(",
        "    model='bge-reranker-v2-m3',",
        "    query=query,",
        "    documents=[{'id': str(i), 'text': doc} for i, doc in enumerate(documents)],",
        "    top_n=3,",
        "    return_documents=True,",
        ")",
        "print(reranked)",
    ),
    md("### 6. Inspect the reranked results"),
    code(
        "def show_reranked_results(query, matches):",
        "    print(f'Query: {query}')",
        "    print()",
        "    for i, m in enumerate(matches):",
        "        print(f'{i+1}. score={m.score:.4f}')",
        "        print(f'   {m.document.text}')",
        "        print()",
        "",
        "",
        "show_reranked_results(query, reranked.data)",
    ),
    md(
        "**Expected behaviour.** The two **Apple-company** documents (iPhone/iPad,",
        "MacBook Pro) should land at the top because the query asks about *Apple's",
        "products*. Apple-fruit documents get lower scores because the model uses",
        "the surrounding context (*products, iPhone, MacBook*) to resolve the",
        "ambiguity — exactly what a reranker is supposed to do.",
    ),

    # ============================================================
    # Part 2: Serverless index setup
    # ============================================================
    md("## Part 2 — Set Up a Serverless Index for Medical Notes"),
    code(
        "!pip install -q pandas torch transformers",
    ),
    md("### 2. Imports + environment settings"),
    code(
        "import os, time",
        "import pandas as pd",
        "from pinecone import Pinecone, ServerlessSpec",
        "from transformers import AutoTokenizer, AutoModel",
        "import torch",
        "",
        "cloud  = os.getenv('PINECONE_CLOUD', 'aws')",
        "region = os.getenv('PINECONE_REGION', 'us-east-1')",
        "",
        "spec = ServerlessSpec(cloud=cloud, region=region)",
        "",
        "index_name = 'medical-notes-index'",
        "print(f'Index will be created at {cloud}/{region} with name {index_name!r}')",
    ),
    md("### 3. Create (or recreate) the index"),
    code(
        "# Clean up any existing index with the same name",
        "if pc.has_index(name=index_name):",
        "    pc.delete_index(name=index_name)",
        "    print('Existing index deleted.')",
        "",
        "# Create a new index — 384-dim matches all-MiniLM-L6-v2; cosine for semantic text similarity",
        "pc.create_index(",
        "    name=index_name,",
        "    dimension=384,",
        "    metric='cosine',",
        "    spec=spec,",
        ")",
        "print('Index created.')",
    ),

    # ============================================================
    # Part 3: Load sample data
    # ============================================================
    md("## Part 3 — Load the Sample Medical Notes Data"),
    code(
        "import requests, tempfile",
        "",
        "SAMPLE_URL = 'https://raw.githubusercontent.com/pinecone-io/examples/refs/heads/master/docs/data/sample_notes_data.jsonl'",
        "",
        "with tempfile.TemporaryDirectory() as tmpdirname:",
        "    file_path = os.path.join(tmpdirname, 'sample_notes_data.jsonl')",
        "    resp = requests.get(SAMPLE_URL, timeout=60)",
        "    resp.raise_for_status()",
        "    with open(file_path, 'wb') as f:",
        "        f.write(resp.content)",
        "    df = pd.read_json(file_path, orient='records', lines=True)",
        "",
        "print('Data shape:', df.shape)",
        "df.head()",
    ),

    # ============================================================
    # Part 4: Upsert
    # ============================================================
    md("## Part 4 — Upsert Data into the Index"),
    code(
        "index = pc.Index(name=index_name)",
        "",
        "index.upsert_from_dataframe(df)",
    ),
    md("### Wait until the index has been populated"),
    code(
        "def is_fresh(index):",
        "    stats = index.describe_index_stats()",
        "    vector_count = stats.total_vector_count",
        "    print('Vector count:', vector_count)",
        "    return vector_count > 0",
        "",
        "",
        "while not is_fresh(index):",
        "    time.sleep(5)",
        "",
        "print('Index ready!')",
        "print(index.describe_index_stats())",
    ),

    # ============================================================
    # Part 5: Embedding + query
    # ============================================================
    md(
        "## Part 5 — Embedding Function and Semantic Search Query",
        "",
        "We embed the clinical question with `sentence-transformers/all-MiniLM-L6-v2`",
        "to match the 384-dimensional space of the indexed notes. We then average",
        "the token embeddings of the last hidden state to get a single sentence",
        "vector.",
    ),
    code(
        "def get_embedding(input_question):",
        "    model_name = 'sentence-transformers/all-MiniLM-L6-v2'",
        "    tokenizer = AutoTokenizer.from_pretrained(model_name)",
        "    model = AutoModel.from_pretrained(model_name)",
        "    encoded_input = tokenizer(",
        "        input_question, padding=True, truncation=True, return_tensors='pt',",
        "    )",
        "    with torch.no_grad():",
        "        model_output = model(**encoded_input)",
        "    # last_hidden_state[0] has shape (seq_len, hidden); average over seq_len = dim 0",
        "    embedding = model_output.last_hidden_state[0].mean(dim=0)",
        "    return embedding",
        "",
        "",
        "question = 'Patient has chest pain and shortness of breath'",
        "query = get_embedding(question).tolist()",
        "",
        "results = index.query(vector=[query], top_k=10, include_metadata=True)",
        "",
        "sorted_matches = sorted(results['matches'], key=lambda x: x['score'], reverse=True)",
        "print(f'Got {len(sorted_matches)} matches.')",
    ),

    # ============================================================
    # Part 6: Display + rerank
    # ============================================================
    md("## Part 6 — Display and Rerank Clinical Notes"),
    md("### 1. Initial search results (vector similarity only)"),
    code(
        "def show_results(question, matches):",
        "    print(f'Question: {question!r}')",
        "    print('\\nResults:')",
        "    for i, match in enumerate(matches):",
        "        print(f'{str(i+1).rjust(4)}. ID: {match[\"id\"]}')",
        "        print(f'      Score   : {match[\"score\"]:.4f}')",
        "        print(f'      Metadata: {match[\"metadata\"]}')",
        "        print()",
        "",
        "",
        "show_results(question, sorted_matches)",
    ),
    md("### 2. Prepare documents for the reranker"),
    code(
        "transformed_documents = [",
        "    {",
        "        'id': match['id'],",
        "        'reranking_field': '; '.join(",
        "            [f'{key}: {value}' for key, value in match['metadata'].items()]",
        "        ),",
        "    }",
        "    for match in results['matches']",
        "]",
        "",
        "print('Sample transformed document:')",
        "print(transformed_documents[0])",
    ),
    md("### 3. Execute serverless reranking with a more specific query"),
    code(
        "refined_query = 'Patient with severe chest pain — possible acute myocardial infarction'",
        "",
        "reranked_results = pc.inference.rerank(",
        "    model='bge-reranker-v2-m3',",
        "    query=refined_query,",
        "    documents=transformed_documents,",
        "    rank_fields=['reranking_field'],",
        "    top_n=3,",
        "    return_documents=True,",
        ")",
    ),
    md("### 4. Show the reranked results"),
    code(
        "def show_reranked_results(question, matches):",
        "    print(f'Question: {question!r}')",
        "    print('\\nReranked Results:')",
        "    for i, match in enumerate(matches):",
        "        print(f'{str(i+1).rjust(4)}. ID: {match.document.id}')",
        "        print(f'      Score          : {match.score:.4f}')",
        "        print(f'      Reranking Field: {match.document.reranking_field}')",
        "        print()",
        "",
        "",
        "show_reranked_results(refined_query, reranked_results.data)",
    ),
    md(
        "**Vector search vs reranking — what changed?** Vector search retrieves the",
        "top-10 notes whose embeddings are closest to the embedding of the original",
        "question. The reranker takes those 10 candidates plus the **refined**",
        "question and re-scores each candidate by *jointly* reading the query and",
        "the candidate's text — which is much more accurate than two independent",
        "embeddings. In a clinical setting that gap is the difference between",
        "showing the on-call resident the right chart first vs the fourth.",
    ),
    md("### 5. Clean up — delete the index to save resources"),
    code(
        "pc.delete_index(name=index_name)",
        "print(f'Index {index_name!r} deleted.')",
    ),

    md(
        "## Summary",
        "",
        "- We ran `bge-reranker-v2-m3` on a toy 5-document set and watched it",
        "  disambiguate **Apple (company)** from **apple (fruit)** based on the",
        "  query context.",
        "- We created a **serverless** Pinecone index (`dimension=384`, `metric='cosine'`),",
        "  loaded sample medical notes from a public GitHub URL, and upserted them",
        "  with `upsert_from_dataframe`.",
        "- We embedded a clinical question with `all-MiniLM-L6-v2`, ran a vector",
        "  search for the top-10 nearest notes, then **reranked** them with",
        "  `bge-reranker-v2-m3` against a refined query.",
        "- The reranker reorders results based on the *joint* relevance of query +",
        "  candidate — typically promoting the most clinically pertinent notes to",
        "  positions 1–3, even when the original embedding search ranked them lower.",
        "",
        "**Why this matters.** In healthcare, a clinician's time is the scarcest",
        "resource. Putting the most relevant note in position 1 instead of position 4",
        "saves seconds per query and minutes per shift — and with high-recall RAG",
        "pipelines, that compounded saving is the difference between a tool people",
        "use and a tool people abandon.",
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
