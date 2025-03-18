import streamlit as st
import pandas as pd
import os
from pathlib import Path
from rag_engine import RAGEngine
from data_processor import DataProcessor

def create_app():
    st.set_page_config(
        page_title="Financial RAG Assistant",
        page_icon="💰",
        layout="wide"
    )
    
    st.title("Financial RAG Assistant")
    st.sidebar.title("Options")
    
    # Initialize the RAG engine
    @st.cache_resource
    def load_rag_engine():
        rag = RAGEngine(
            model_path="models/ggml-model-q4_0.bin",
            data_path="data/processed_data.csv"
        )
        return rag
    
    # Check if processed data exists, otherwise create sample data
    data_file = Path("data/processed_data.csv")
    if not data_file.exists():
        st.sidebar.warning("No processed data found. Creating sample data...")
        processor = DataProcessor()
        processor.create_sample_data()
        st.sidebar.success("Sample data created!")
    
    # Try to load the RAG engine
    try:
        rag_engine = load_rag_engine()
        st.sidebar.success(f"Loaded {len(rag_engine.df_chunks)} chunks of financial data")
    except Exception as e:
        st.error(f"Error loading RAG engine: {e}")
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
        min_value=1,
        max_value=10,
        value=5,
        help="Number of document chunks to retrieve for each query"
    )
    
    # Show advanced options
    show_details = st.sidebar.checkbox("Show retrieval details", value=False)
    
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
                result = rag_engine.process_query(
                    query=query,
                    k=k_chunks,
                    retrieval_method=retrieval_method
                )
                
                # Display if the query is valid
                if not result["is_valid"]:
                    st.warning(result["reason"])
                    st.info(result["answer"])
                    st.stop()
                
                # Display the answer
                st.markdown("### Answer")
                st.write(result["answer"])
                
                # Display retrieved contexts if show_details is enabled
                if show_details and result["contexts"]:
                    st.markdown("### Relevant Information")
                    for i, context in enumerate(result["contexts"]):
                        with st.expander(f"Source {i+1}"):
                            st.write(context)
                            
                            # Get metadata for this chunk
                            chunk_id = result["metadata"]["retrieval_results"][i]["chunk_id"]
                            metadata = rag_engine.df_chunks[rag_engine.df_chunks["chunk_id"] == chunk_id].iloc[0]
                            
                            # Display metadata
                            st.markdown(f"**Source**: {metadata['source_file']}")
                            st.markdown(f"**Year**: {metadata['year']}")
                            
                            # Show score if available
                            if "score" in result["metadata"]["retrieval_results"][i]:
                                score = result["metadata"]["retrieval_results"][i]["score"]
                                st.markdown(f"**Relevance Score**: {score:.4f}")
    
    # Display data information
    with st.sidebar.expander("Data Information"):
        if rag_engine.df_chunks is not None:
            st.write(f"Total chunks: {len(rag_engine.df_chunks)}")
            
            # Count by year
            year_counts = rag_engine.df_chunks['year'].value_counts()
            st.write("Data by year:")
            for year, count in year_counts.items():
                st.write(f"- {year}: {count} chunks")
            
            # Count by document type
            doc_counts = rag_engine.df_chunks['document_type'].value_counts()
            st.write("Data by document type:")
            for doc_type, count in doc_counts.items():
                st.write(f"- {doc_type}: {count} chunks")

if __name__ == "__main__":
    create_app() 