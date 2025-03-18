# Financial RAG - Quick Start Guide

This guide will help you quickly get started with the Financial RAG system for answering questions about financial statements.

## Prerequisites

- Python 3.8 or later
- Financial statements in PDF format (annual reports, 10-K forms, etc.)
- (Optional) A quantized LLM model in GGML format

## Installation

1. **Clone or download this repository**

2. **Install dependencies**

```bash
cd financial_rag
pip install -r requirements.txt
```

3. **Run the application**

```bash
python run.py
```

This will start the Streamlit application and open it in your default web browser.

## Using the Application

### Step 1: Upload Financial Documents

1. In the sidebar, click "Browse files" under "Upload Financial Statements"
2. Select one or more PDF files containing financial statements
3. Click "Process Uploaded Files" to extract and chunk the text

### Step 2: Initialize the Model

1. Under "Model Settings" in the sidebar:
   - Choose whether to use GPU (if available)
   - Select an embedding model (default is fine for most uses)
   - Upload your own LLM model (optional) or use the default

2. Click "Initialize/Reset Model" to prepare the system

### Step 3: Ask Questions

1. In the main area, enter your financial question in the text box
2. Click "Submit" to process your question
3. Review the answer and confidence score
4. Explore retrieved contexts by expanding the "View Retrieved Contexts" section

## Example Questions

Here are some examples of questions you can ask:

- "What was the total revenue in 2022?"
- "How did gross margin change from 2021 to 2022?"
- "What were the major expenses reported in the last fiscal year?"
- "What is the debt-to-equity ratio based on the balance sheet?"
- "Did the company report a profit or loss in 2022?"

## Advanced Configuration

The sidebar provides additional settings:

- **Guardrail Settings**: Toggle input and output guardrails
- **Advanced Settings**: Adjust retrieval parameters and strategies

## Troubleshooting

- **Missing data error**: Ensure you've uploaded and processed financial statements
- **Model initialization errors**: Check if you have sufficient memory for the LLM
- **Slow performance**: Consider using a smaller model or enabling GPU if available

## Next Steps

- Try different financial questions to test the system
- Experiment with different retrieval and merging strategies
- Upload different financial statements to compare results 