import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import os
import re
from sentence_transformers import SentenceTransformer, util
import faiss
from rank_bm25 import BM25Okapi
from langchain_community.llms import LlamaCpp
import time
from pathlib import Path

class InputGuardrail:
    """Simple input guardrail to validate financial queries."""
    
    def __init__(self):
        self.financial_terms = [
            "revenue", "profit", "loss", "income", "expense", "asset", "liability",
            "balance", "sheet", "statement", "cash", "flow", "dividend", "equity",
            "debt", "earnings", "ebitda", "margin", "ratio", "financial", "fiscal",
            "company", "companies", "business", "performance", "year", "quarter",
            "annual", "quarterly", "growth", "increase", "decrease", "report"
        ]
    
    def validate_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate if the query is finance-related.
        
        Returns:
            (is_valid, reason) tuple
        """
        query_lower = query.lower()
        
        # Check if query has any financial terms
        has_financial_terms = any(term in query_lower for term in self.financial_terms)
        
        if not has_financial_terms:
            return False, "Query does not appear to be related to financial information."
        
        return True, None
    
    def suggest_reformulation(self, query: str) -> str:
        """Suggest how to reformulate an invalid query."""
        return ("I can only answer questions about financial information. "
                "Please ask a question related to company financials, such as "
                "revenue, profit, expenses, or financial performance.")


class RAGEngine:
    def __init__(self, 
                 model_path: str = "models/ggml-model-q4_0.bin",
                 data_path: str = "data/processed_data.csv",
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 use_gpu: bool = False):
        """
        Initialize the RAG engine.
        
        Args:
            model_path: Path to the LLM model (quantized GGML format)
            data_path: Path to the processed financial data CSV
            embedding_model_name: Name of the embedding model to use
            use_gpu: Whether to use GPU for LLM inference
        """
        self.model_path = model_path
        self.data_path = data_path
        self.embedding_model_name = embedding_model_name
        self.use_gpu = use_gpu
        
        # Initialize components
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.bm25_tokenized_corpus = None
        self.bm25 = None
        self.llm = None
        
        # Initialize guardrails
        self.input_guardrail = InputGuardrail()
        
        # Data storage
        self.df_chunks = None
        self.index = None
        self.chunk_ids = None
        
        # Load data if available
        if os.path.exists(data_path):
            self.load_data(data_path)
        
    def load_data(self, data_path: str):
        """Load processed financial data."""
        self.df_chunks = pd.read_csv(data_path)
        print(f"Loaded {len(self.df_chunks)} chunks from {data_path}")
        
        # Build index and BM25
        self.build_index()
        self.initialize_bm25()
        
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for BM25 indexing."""
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenize (simple whitespace tokenization)
        tokens = text.split()
        
        # Filter out very short tokens
        tokens = [token for token in tokens if len(token) > 1]
        
        return tokens
    
    def initialize_bm25(self):
        """Initialize BM25 with the current corpus."""
        if self.df_chunks is None or len(self.df_chunks) == 0:
            return
        
        # Preprocess and tokenize each document
        corpus = self.df_chunks['text'].tolist()
        self.bm25_tokenized_corpus = [self.preprocess_text(doc) for doc in corpus]
        
        # Create BM25 index
        self.bm25 = BM25Okapi(self.bm25_tokenized_corpus)
        print("BM25 index initialized")
    
    def build_index(self):
        """Build the embedding index from the loaded data."""
        if self.df_chunks is None:
            return
        
        print("Building embedding index...")
        start_time = time.time()
        
        # Get texts and chunk IDs
        texts = self.df_chunks['text'].tolist()
        self.chunk_ids = self.df_chunks['chunk_id'].tolist()
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        vector_dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(vector_dimension)
        
        # Add vectors to the index
        embeddings = embeddings.astype(np.float32)  # Convert to float32 for FAISS
        self.index = faiss.IndexIDMap(self.index)
        self.index.add_with_ids(embeddings, np.array(range(len(embeddings))))
        
        print(f"Index built in {time.time() - start_time:.2f} seconds")
    
    def load_llm(self):
        """Load the language model for generation."""
        if not os.path.exists(self.model_path):
            print(f"LLM model not found at {self.model_path}")
            return False
            
        try:
            # Try to use LlamaCpp for efficient inference
            n_gpu_layers = 1 if self.use_gpu else 0
            n_batch = 512
            
            self.llm = LlamaCpp(
                model_path=self.model_path,
                n_gpu_layers=n_gpu_layers,
                n_batch=n_batch,
                n_ctx=2048,
                f16_kv=self.use_gpu,  # Only for GPU acceleration
                verbose=False
            )
            print(f"Loaded LLM: {self.model_path}")
            return True
        except Exception as e:
            print(f"Failed to load LLM: {e}")
            return False
    
    def embedding_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar texts using vector embeddings."""
        if self.index is None:
            return []
            
        # Encode the query
        query_embedding = self.embedding_model.encode([query])[0].reshape(1, -1).astype(np.float32)
        
        # Search in the FAISS index
        distances, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx >= 0:  # Some indices might be -1 if less than k results found
                score = 1.0 / (1.0 + distances[0][i])  # Convert distance to similarity score
                results.append({
                    "chunk_id": int(idx),
                    "score": float(score)
                })
        
        return results
    
    def bm25_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search using BM25 algorithm."""
        if self.bm25 is None:
            return []
        
        # Preprocess the query
        tokenized_query = self.preprocess_text(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k document indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        # Format results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include documents with positive scores
                results.append({
                    "chunk_id": int(idx),
                    "score": float(scores[idx])
                })
        
        return results
        
    def hybrid_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Combine vector search and BM25 for better retrieval."""
        # Get results from both methods
        embedding_results = self.embedding_search(query, k)
        bm25_results = self.bm25_search(query, k)
        
        # Create dictionaries for easy lookup
        embedding_scores = {result["chunk_id"]: result["score"] for result in embedding_results}
        bm25_scores = {result["chunk_id"]: result["score"] for result in bm25_results}
        
        # Combine and normalize scores
        all_chunk_ids = set(list(embedding_scores.keys()) + list(bm25_scores.keys()))
        
        # Normalize scores
        max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0
        max_embedding = max(embedding_scores.values()) if embedding_scores else 1.0
        
        # Calculate hybrid scores (equal weights by default)
        weights = (0.5, 0.5)  # (bm25_weight, embedding_weight)
        
        hybrid_results = []
        for chunk_id in all_chunk_ids:
            # Default to 0 if score is missing
            bm25_score = bm25_scores.get(chunk_id, 0) / max_bm25
            embedding_score = embedding_scores.get(chunk_id, 0) / max_embedding
            
            # Calculate weighted hybrid score
            hybrid_score = weights[0] * bm25_score + weights[1] * embedding_score
            
            hybrid_results.append({
                "chunk_id": chunk_id,
                "score": hybrid_score
            })
        
        # Sort by hybrid score and take top k
        hybrid_results = sorted(hybrid_results, key=lambda x: x["score"], reverse=True)[:k]
        
        return hybrid_results
    
    def retrieve(self, query: str, k: int = 5, method: str = "hybrid") -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query using different methods.
        
        Args:
            query: User query
            k: Number of chunks to retrieve
            method: Retrieval method ('embeddings', 'bm25', or 'hybrid')
            
        Returns:
            List of retrieval results with chunk_id and score
        """
        if method == "embeddings":
            return self.embedding_search(query, k)
        elif method == "bm25":
            return self.bm25_search(query, k)
        else:  # hybrid is the default
            return self.hybrid_search(query, k)
    
    def get_contexts(self, results: List[Dict[str, Any]]) -> List[str]:
        """Get context texts from chunk IDs."""
        contexts = []
        
        for result in results:
            chunk_id = result["chunk_id"]
            chunk_text = self.df_chunks[self.df_chunks['chunk_id'] == chunk_id]['text'].iloc[0]
            contexts.append(chunk_text)
            
        return contexts
    
    def format_prompt(self, query: str, contexts: List[str]) -> str:
        """Format the prompt for the LLM with the query and contexts."""
        # Join contexts with separators
        context_text = "\n\n---\n\n".join([f"Context {i+1}:\n{context}" for i, context in enumerate(contexts)])
        
        # Create the prompt template
        prompt_template = """You are a helpful financial assistant that provides accurate information based only on the given contexts. 
If the answer cannot be found in the contexts, say that you don't have enough information.
Do not make up information that is not in the contexts.

CONTEXTS:
{context_text}

USER QUESTION: {query}

ANSWER:"""
        
        # Format the prompt
        prompt = prompt_template.format(context_text=context_text, query=query)
        
        return prompt
    
    def generate_answer(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate an answer using the LLM."""
        if self.llm is None:
            if not self.load_llm():
                return "LLM model not available. Unable to generate response."
        
        try:
            return self.llm.invoke(prompt, max_tokens=max_tokens)
        except Exception as e:
            return f"Error generating response: {e}"
    
    def process_query(self, query: str, k: int = 5, retrieval_method: str = "hybrid") -> Dict[str, Any]:
        """
        Process a query through the entire RAG pipeline.
        
        Args:
            query: User query
            k: Number of chunks to retrieve
            retrieval_method: Method for retrieval ('embeddings', 'bm25', or 'hybrid')
            
        Returns:
            Dict containing the results, contexts, and metadata
        """
        result = {
            "query": query,
            "is_valid": True,
            "reason": None,
            "answer": None,
            "contexts": [],
            "metadata": {}
        }
        
        # Apply input guardrail
        is_valid, reason = self.input_guardrail.validate_query(query)
        result["is_valid"] = is_valid
        result["reason"] = reason
            
        if not is_valid:
            result["answer"] = self.input_guardrail.suggest_reformulation(query)
            return result
        
        # Retrieve relevant chunks
        retrieval_results = self.retrieve(
            query=query,
            k=k,
            method=retrieval_method
        )
        
        result["metadata"]["retrieval_method"] = retrieval_method
        result["metadata"]["retrieval_results"] = retrieval_results
        
        if not retrieval_results:
            result["answer"] = "I couldn't find any relevant information to answer your question."
            return result
        
        # Get contexts
        contexts = self.get_contexts(retrieval_results)
        result["contexts"] = contexts
        
        # Format prompt and generate answer
        prompt = self.format_prompt(query, contexts)
        result["metadata"]["prompt"] = prompt
        
        # Generate answer
        result["answer"] = self.generate_answer(prompt)
            
        return result

if __name__ == "__main__":
    # Example usage
    rag = RAGEngine()
    
    # Process a sample query
    result = rag.process_query("What was the company's revenue in 2023?")
    
    print(f"Answer: {result['answer']}") 