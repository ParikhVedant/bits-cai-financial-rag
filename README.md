# Financial RAG Assistant

A Retrieval-Augmented Generation (RAG) system for financial question answering using company annual reports.

## Overview

This system allows users to ask questions about financial data extracted from annual reports. It uses a combination of:

- **Embedding-based retrieval**: Using sentence-transformers to find semantically similar content
- **Keyword-based retrieval**: Using BM25 for term-frequency based matching
- **Open-source LLM**: Using LlamaCpp for inference on a small language model
- **Input validation**: Ensuring queries are related to financial information

## Setup

1. Create a conda environment:
```bash
conda create -n financial_rag python=3.9
conda activate financial_rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download an open-source LLM model (GGML format):
- Create a `models` directory in the project root
- Download a compatible model like Llama-2-7b-GGML or similar
- Rename it to `ggml-model-q4_0.bin` and place it in the `models` directory

4. Process your financial documents:
- Place PDF financial documents in the `data` directory
- Run the data processor to extract and chunk the information:
```bash
python data_processor.py
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Enter financial questions in the input field
- Ask about revenue, profit, assets, etc.
- Try different retrieval methods to compare results

3. View the results, including:
- The generated answer
- Retrieved contexts (optional)
- Relevance scores for each chunk

## Components

- `data_processor.py`: Extracts text from PDFs and creates chunks
- `rag_engine.py`: Core RAG functionality including embedding, retrieval, and generation
- `app.py`: Streamlit user interface

## Features

- **Hybrid Search**: Combines embeddings and BM25 for better retrieval
- **Input Guardrails**: Validates that queries are finance-related
- **Interactive UI**: Allows experimenting with different retrieval parameters
- **Document Context**: Shows sources of information used to generate answers

## Test Queries

Try these example queries:
1. "What was the company's revenue in 2023?" (high confidence)
2. "How did the operating expenses change from 2022 to 2023?" (medium confidence)
3. "What is the capital of France?" (should be rejected by guardrails) 