import streamlit as st
import pandas as pd
import os
from pathlib import Path
from openai_assistant import OpenAIAssistant

def create_app():
    st.set_page_config(
        page_title="Financial RAG Assistant",
        page_icon="💰",
        layout="wide"
    )
    
    st.title("Financial RAG Assistant")
    
    # Add group members information
    st.sidebar.markdown("## Group Members Name with Student ID:")
    st.sidebar.markdown("1. PARIKH VEDANT ASHISH ALPA 2023AA05369")
    st.sidebar.markdown("2. KANSARA HARSH BHARAT BHAVINI 2023AA05351")
    
    # Separator
    st.sidebar.markdown("---")
    
    # Handle URL forwarding - display same content regardless of URL path
    # This ensures the app works the same whether accessed at root, /cai, or any other path
    
    st.sidebar.title("Options")
    
    # Initialize the OpenAI Assistant
    @st.cache_resource
    def load_assistant():
        assistant = OpenAIAssistant()
        return assistant
    
    # Try to load the OpenAI Assistant
    try:
        assistant = load_assistant()
    except Exception as e:
        st.error(f"Error loading OpenAI Assistant: {e}")
        st.stop()
    
    # Retrieval method selection
    retrieval_method = st.sidebar.selectbox(
        "Retrieval Method",
        ["hybrid", "embeddings", "bm25"],
        index=0,
        help="Method used to retrieve relevant document chunks"
    )
    
    # Number of chunks to retrieve
    k_chunks = st.sidebar.slider(
        "Number of chunks to retrieve",
        min_value=5,
        max_value=10,
        value=5,
        help="Number of document chunks to retrieve for each query"
    )
    
    # Add reset session button
    if st.sidebar.button("Reset Session"):
        # Create a new thread
        assistant.create_thread()
        st.sidebar.success("Session reset! Started a new conversation.")
    
    # Query input
    query = st.text_input(
        "Ask a question about the company's financial performance:",
        placeholder="e.g., What was the revenue in 2023?"
    )
    
    if st.button("Submit"):
        if not query:
            st.warning("Please enter a question")
        else:
            with st.spinner("Searching for information..."):
                # Process the query
                result = assistant.process_query(
                    query=query,
                    k=k_chunks,
                    retrieval_method=retrieval_method
                )
                
                # Display the answer
                st.markdown("### Answer")
                st.write(result["answer"])
                
                # Show thread ID in the sidebar for debugging
                st.sidebar.markdown("### Current Session")
                st.sidebar.text(f"Thread ID: {result['metadata'].get('thread_id', 'N/A')}")

if __name__ == "__main__":
    create_app() 