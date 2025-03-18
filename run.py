#!/usr/bin/env python
import os
import sys
import streamlit.web.cli as stcli
from pathlib import Path

def check_data():
    """Check if data directory and processed data exist."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    processed_file = data_dir / "processed_data.csv"
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    if not processed_file.exists():
        print("\033[93m⚠️  No processed data found.\033[0m")
        print("Either:")
        print("1. Place PDF files in the 'data' directory and run 'python data_processor.py', or")
        print("2. Sample data will be created automatically when running the app")
    
    model_file = model_dir / "ggml-model-q4_0.bin"
    if not model_file.exists():
        print("\033[93m⚠️  LLM model not found at models/ggml-model-q4_0.bin\033[0m")
        print("For full functionality, please:")
        print("1. Download a compatible GGML model (e.g., Llama-2-7b)")
        print("2. Place it in the 'models' directory as 'ggml-model-q4_0.bin'")
        print("Note: The system will still run with limited functionality.")

def run_app():
    """Run the Streamlit app."""
    check_data()
    print("\033[92m🚀 Starting the Financial RAG Assistant...\033[0m")
    sys.argv = ["streamlit", "run", "app.py"]
    sys.exit(stcli.main())

if __name__ == "__main__":
    run_app() 