import os
import re
import PyPDF2
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path

class DataProcessor:
    def __init__(self, data_dir: str = "data"):
        """Initialize the data processor with the directory containing financial documents."""
        self.data_dir = data_dir
        
    def load_pdf(self, file_path: str) -> str:
        """Load and extract text from a PDF file."""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
        return text
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess the extracted text."""
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters and formatting issues
        text = re.sub(r'[^\w\s.,;:()\-$%]', '', text)
        return text.strip()
    
    def split_into_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into chunks of specified size with overlap."""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            # If this is not the last chunk, try to find a space to break at
            if end < text_length:
                # Try to find a space to break at
                while end > start + chunk_size - 100 and text[end] != ' ':
                    end -= 1
                if end == start + chunk_size - 100:  # If we couldn't find a space, just use the chunk_size
                    end = start + chunk_size
            
            # Extract the chunk
            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            # Move start for next chunk, with overlap
            start = end - overlap
            if start < 0 or start >= text_length:
                break
                
        return chunks
    
    def extract_metadata(self, text: str, file_name: str) -> Dict[str, Any]:
        """Extract metadata from the financial document."""
        # Extract year from filename
        year_match = re.search(r'20\d{2}', file_name)
        year = year_match.group(0) if year_match else "Unknown"
        
        # Simple document type detection
        doc_type = "Annual Report"
        if "income statement" in text.lower():
            doc_type = "Income Statement"
        elif "balance sheet" in text.lower():
            doc_type = "Balance Sheet"
        elif "cash flow" in text.lower():
            doc_type = "Cash Flow Statement"
            
        return {
            "year": year,
            "document_type": doc_type,
            "file_name": file_name
        }
    
    def process_documents(self, chunk_size: int = 1000, overlap: int = 200) -> pd.DataFrame:
        """Process all financial documents in the data directory."""
        chunk_id = 0
        data = []
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.pdf'):
                file_path = os.path.join(self.data_dir, filename)
                print(f"Processing: {filename}")
                
                try:
                    # Extract and clean text
                    raw_text = self.load_pdf(file_path)
                    cleaned_text = self.clean_text(raw_text)
                    
                    # Extract metadata
                    metadata = self.extract_metadata(cleaned_text, filename)
                    
                    # Split into chunks
                    chunks = self.split_into_chunks(cleaned_text, chunk_size, overlap)
                    
                    # Add chunks to data list
                    for chunk in chunks:
                        data.append({
                            "chunk_id": chunk_id,
                            "text": chunk,
                            "year": metadata["year"],
                            "document_type": metadata["document_type"],
                            "source_file": metadata["file_name"]
                        })
                        chunk_id += 1
                    
                    print(f"  Added {len(chunks)} chunks from {filename}")
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        # Create DataFrame from data
        df = pd.DataFrame(data)
        return df
    
    def save_processed_data(self, df: pd.DataFrame, output_file: str = "data/processed_data.csv"):
        """Save processed data to a CSV file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"Processed data saved to {output_file}")
        return output_file
    
    def create_sample_data(self, output_file: str = "data/processed_data.csv"):
        """Create sample data from financial statements."""
        # Sample financial data for demo purposes
        financial_data = [
            {
                "chunk_id": 0,
                "text": "Annual Report 2023-24 Jio Financial Services Limited. Revenues for FY 2023-24 increased to ₹1,342 crore, representing a growth of 15% over the previous year.",
                "year": "2023",
                "document_type": "Annual Report",
                "source_file": "annual-report-2023-2024.pdf",
            },
            {
                "chunk_id": 1,
                "text": "Net profit for FY 2023-24 stood at ₹560 crore, compared to ₹487 crore in the previous year. This represents a year-on-year growth of 15%.",
                "year": "2023",
                "document_type": "Annual Report", 
                "source_file": "annual-report-2023-2024.pdf",
            },
            {
                "chunk_id": 2,
                "text": "Total assets as of March 31, 2024 were ₹7,890 crore, compared to ₹6,570 crore as of March 31, 2023, representing a 20% increase year-over-year.",
                "year": "2023",
                "document_type": "Annual Report",
                "source_file": "annual-report-2023-2024.pdf",
            },
            {
                "chunk_id": 3,
                "text": "The company's dividend payout for FY 2023-24 was ₹15 per share, representing a dividend yield of 2.5% based on the closing share price on March 31, 2024.",
                "year": "2023",
                "document_type": "Annual Report",
                "source_file": "annual-report-2023-2024.pdf",
            },
            {
                "chunk_id": 4,
                "text": "During FY 2022-23, the company reported revenues of ₹1,167 crore and a net profit of ₹487 crore.",
                "year": "2022",
                "document_type": "Annual Report",
                "source_file": "annual-report-2022-2023.pdf",
            },
            {
                "chunk_id": 5,
                "text": "The operating expenses for FY 2022-23 were ₹580 crore, representing 49.7% of total revenue compared to 52.3% in the previous year.",
                "year": "2022",
                "document_type": "Annual Report",
                "source_file": "annual-report-2022-2023.pdf",
            },
            {
                "chunk_id": 6,
                "text": "The company's return on equity (ROE) for FY 2022-23 was 12.8%, compared to 11.5% in the previous year, showing improved profitability.",
                "year": "2022",
                "document_type": "Annual Report",
                "source_file": "annual-report-2022-2023.pdf",
            },
        ]
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Create a DataFrame
        df = pd.DataFrame(financial_data)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Sample data created and saved to {output_file}")
        
        return df

if __name__ == "__main__":
    processor = DataProcessor()
    
    # Try to process documents if available
    try:
        df = processor.process_documents()
        processor.save_processed_data(df)
    except Exception as e:
        print(f"Error processing documents: {e}")
        print("Creating sample data instead...")
        df = processor.create_sample_data() 