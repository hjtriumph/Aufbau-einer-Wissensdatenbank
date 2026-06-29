# Building Technology Knowledge Base Using LLMs

This repository contains the implementation for a thesis project on building an intelligent knowledge base for building technology documents using large language models (LLMs), LangChain, Chroma, and retrieval-augmented generation (RAG).

The system ingests research papers and technical documents, stores them in a persistent vector database, and answers natural-language questions with evidence from the retrieved document context. The project focuses on building technology topics such as HVAC systems, energy efficiency, heating-load forecasting, occupancy-aware prediction, missing-data imputation, and explainable AI for building performance analysis.

## Project Overview

The main goal is to build and evaluate a searchable, context-aware question answering system for building technology literature.

The project supports:

- PDF, TXT, and Markdown document ingestion
- Text chunking with configurable chunk size and overlap
- Local sentence-transformer embeddings
- Persistent Chroma vector storage
- RAG-based question answering with source/page/chunk evidence
- Few-shot RAG prompting for answer-quality comparison
- Direct LLM baseline without retrieval
- Benchmark scripts for completeness, key-point coverage, answer relevance, and latency
- Radar-chart visualization of benchmark results
- Experimental ReAct-style agentic RAG graph using LangGraph

## Repository Structure

```text
.
|-- data/
|   `-- docs/                         # Source papers and technical documents
|-- storage/
|   `-- chroma/                       # Persistent Chroma vector database
|-- src/
|   |-- config.py                     # Environment/configuration loading
|   |-- ingest.py                     # Document loading, chunking, embedding, vector-store creation
|   |-- rag_chain.py                  # Standard RAG pipeline
|   |-- few_shot_chain.py             # Few-shot RAG pipeline with structured answers
|   |-- few_shot_no_cot.py            # Few-shot RAG variant with short direct answers
|   |-- cli.py                        # Command-line interface for LLM, RAG, and few-shot QA
|   |-- agent_graph.py                # Experimental LangGraph ReAct-style agentic RAG
|   |-- test_retriever.py             # Retrieval debugging script
|   |-- benchmark_cases.py            # Hand-written benchmark questions and reference answers
|   |-- benchmark_qa.py               # Main QA benchmark
|   |-- benchmark_qa_llm_vs_1fewshot.py
|   |-- benchmark_fewshot_cot_compare.py
|   |-- visualize_radar.py
|   |-- visualize_radar_llm_vs_1fewshot.py
|   `-- visualize_fewshot_cot_radar.py
|-- benchmark_outputs_qa/             # Benchmark outputs for LLM vs multiple few-shot settings
|-- benchmark_outputs_qa_llm_vs_1fewshot/
|-- benchmark_outputs_fewshot_cot_compare/
|-- requirements.txt
|-- settings.json
`-- README.md
```

## Data

Place source documents in:

```text
data/docs/
```

The current document collection contains research papers related to:

- heating-load and cooling-load prediction
- occupancy data for building energy forecasting
- explainable AI and SHAP analysis
- missing heating-load data imputation
- transformer and Mixture-of-Experts forecasting models
- feature importance and building characteristics

Supported input formats are:

- `.pdf`
- `.txt`
- `.md`

## Environment Setup

Create and activate a Python environment. The original project was developed with a Conda environment named `chain_1`.

```bash
conda create -n chain_1 python=3.12
conda activate chain_1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If your installed LangChain version raises import errors for Chroma or HuggingFace embeddings, install the split packages as well:

```bash
pip install langchain-chroma langchain-huggingface
```

For benchmark visualization, the radar scripts also require:

```bash
pip install pandas matplotlib
```

## Configuration

Create a `.env` file in the project root. The code uses DeepSeek through the OpenAI-compatible `ChatOpenAI` interface.

Example:

```env
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

DOCS_DIR=./data/docs
CHROMA_DIR=./storage/chroma
COLLECTION_NAME=building_tech_kb

TEMPERATURE=0.2
MAX_RETRIES=2
K=6
FETCH_K=20
SEARCH_TYPE=mmr
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
```

Important configuration values:

- `DEEPSEEK_API_KEY`: API key for the LLM provider
- `DEEPSEEK_BASE_URL`: OpenAI-compatible API endpoint
- `DEEPSEEK_MODEL`: chat model name
- `EMBEDDING_MODEL`: local sentence-transformer embedding model
- `DOCS_DIR`: document folder
- `CHROMA_DIR`: persistent vector database folder
- `SEARCH_TYPE`: `mmr` or `similarity`
- `K`: number of retrieved chunks returned to the QA chain
- `FETCH_K`: candidate pool size for MMR retrieval
- `CHUNK_SIZE` and `CHUNK_OVERLAP`: document splitting parameters

## Build the Knowledge Base

Run ingestion after placing documents in `data/docs/`.

```bash
python -m src.ingest
```

This step:

1. Loads PDF, TXT, and Markdown files.
2. Splits documents into chunks.
3. Adds metadata such as source file, page, and chunk id.
4. Embeds chunks using the configured sentence-transformer model.
5. Writes the persistent Chroma database to `storage/chroma/`.

## Test Retrieval

Use the retriever test script to inspect which chunks are returned for a query.

```bash
python -m src.test_retriever --q "HVAC system"
```

JSON output is also supported:

```bash
python -m src.test_retriever --q "heating load forecasting occupancy" --json
```

## Ask Questions

The main CLI supports three modes:

- `llm`: direct LLM answer without retrieval
- `rag`: standard retrieval-augmented answer
- `fewshot`: few-shot RAG answer using examples from the building technology domain

### Direct LLM baseline

```bash
python -m src.cli --mode llm --q "Explain VAV control."
```

### Standard RAG

```bash
python -m src.cli --mode rag --q "According to the papers, does adding occupancy improve heating forecasting accuracy?"
```

### Few-shot RAG

```bash
python -m src.cli --mode fewshot --fewshot-n 3 --q "Why are MAE, NRMSE, and sMAPE all reported instead of just one metric?"
```

### JSON output

```bash
python -m src.cli --mode fewshot --fewshot-n 3 --q "What input features are used in the Mixture of Experts forecasting task?" --json
```

### Show sources

```bash
python -m src.cli --mode fewshot --fewshot-n 3 --q "How is explainability achieved in the MoE framework?" --json --show-sources
```

## Benchmarking

Benchmark questions and reference answers are defined in `src/benchmark_cases.py`.

The main benchmark compares:

- `llm`
- `fewshot_0`
- `fewshot_1`
- `fewshot_3`
- `fewshot_5`
- `fewshot_7`
- `fewshot_10`

Run the benchmark on the first 10 cases:

```bash
python -m src.benchmark_qa --num-cases 10
```

Run all cases:

```bash
python -m src.benchmark_qa --num-cases 0
```

Run selected methods only:

```bash
python -m src.benchmark_qa --methods llm fewshot_1 fewshot_7 --num-cases 10
```

Benchmark outputs are saved to `benchmark_outputs_qa/`:

- `benchmark_raw.jsonl`
- `benchmark_raw.csv`
- `benchmark_summary_overall.csv`
- `benchmark_summary_overall.json`
- `benchmark_summary_radar.csv`
- `benchmark_summary_radar.json`

The benchmark evaluates:

- `completeness`: judged coverage of the reference answer
- `key_point_coverage`: string-based coverage of manually defined key points
- `answer_relevance`: judged relevance to the question
- `latency_sec`: response time per method and case

## Visualization

Generate a radar chart from benchmark results:

```bash
python -m src.visualize_radar
```

Default output:

```text
benchmark_outputs_qa/radar_chart_qa.png
```

Other comparison scripts are available for specific experiments:

```bash
python -m src.benchmark_qa_llm_vs_1fewshot --num-cases 10
python -m src.visualize_radar_llm_vs_1fewshot

python -m src.benchmark_fewshot_cot_compare --num-cases 10
python -m src.visualize_fewshot_cot_radar
```

## Agentic RAG Experiment

`src/agent_graph.py` contains an experimental ReAct-style agent built with LangGraph.

The graph follows this loop:

```text
think -> search -> observe -> think -> ... -> final_answer
```

The LLM decides whether to search the knowledge base again or finish with a final answer. This module is useful for experimenting with multi-step retrieval, but the current command-line interface in `src/cli.py` exposes only `llm`, `rag`, and `fewshot` modes.

## Example Workflow

```bash
conda activate chain_1
pip install -r requirements.txt
python -m src.ingest
python -m src.test_retriever --q "occupancy heating load forecasting"
python -m src.cli --mode fewshot --fewshot-n 3 --q "Does adding occupancy data improve heating and electricity load forecasting in hotels?"
python -m src.benchmark_qa --num-cases 10
python -m src.visualize_radar
```

## Notes

- Do not commit real API keys in `.env`.
- Re-run `python -m src.ingest` after adding, removing, or changing documents in `data/docs/`.
- If benchmark results look weak, first inspect retrieval quality with `src.test_retriever`.
- If the answer says `INSUFFICIENT_CONTEXT`, the relevant evidence was not retrieved or is not present in the indexed documents.
- The project currently uses a DeepSeek OpenAI-compatible chat model, but the same LangChain interface can be adapted to other OpenAI-compatible providers.
