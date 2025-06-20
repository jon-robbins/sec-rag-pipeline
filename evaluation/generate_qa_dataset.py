#!/usr/bin/env python3
"""
Generate a balanced QA dataset from SEC filing chunks.

This script:
1. Loads chunks from data/chunks.pkl
2. Creates a stratified sample balanced across companies, years, sections, and chunk sizes
3. Generates QA pairs using an OpenAI model
4. Saves the results to data/qa_dataset.jsonl

Usage:
    python evaluation/generate_qa_dataset.py
"""

import os
import sys
import pickle
import json
import random
import tiktoken
import time
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Any
from rag.config import QA_DATASET_PATH
from pprint import pprint
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# --- Core Logic for Sampling and QA Generation ---

ENCODER = tiktoken.encoding_for_model("gpt-4")

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    return len(ENCODER.encode(text))

def classify_chunk_by_tokens(text: str) -> str:
    """Classify chunk as short, medium, or large based on token count."""
    tokens = count_tokens(text)
    if tokens <= 350: return "short"
    if tokens <= 500: return "medium"
    return "large"

class BalancedChunkSampler:
    """Creates a balanced, stratified sample of chunks."""
    def __init__(self, max_per_group: int = 3, balance_companies: bool = True):
        self.max_per_group = max_per_group
        self.balance_companies = balance_companies

    def group_chunks_by_keys(self, chunks: List) -> Dict:
        """Group chunks by (ticker, fiscal_year, section, chunk_class)."""
        grouped = defaultdict(list)
        for chunk in chunks:
            # Handle both dict and object formats
            if isinstance(chunk, dict):
                metadata = chunk["metadata"]
                text = chunk["text"]
            else:
                metadata = chunk.metadata
                text = chunk.text
            
            # Handle both old ('item') and new ('section') field names
            section = metadata.get("section") or metadata.get("item")
                
            key = (
                metadata["ticker"],
                metadata["fiscal_year"],
                section,
                classify_chunk_by_tokens(text)
            )
            grouped[key].append(chunk)
        return grouped

    def stratified_sample(self, grouped_chunks: Dict) -> List:
        """Sample chunks evenly from each group."""
        sampled = []
        for group_chunks in grouped_chunks.values():
            sample_size = min(len(group_chunks), self.max_per_group)
            sampled.extend(random.sample(group_chunks, sample_size))
        
        return self._balance_by_company(sampled) if self.balance_companies else sampled
    
    def _balance_by_company(self, chunks: List) -> List:
        """Ensure equal representation across companies."""
        company_chunks = defaultdict(list)
        for chunk in chunks:
            # Handle both dict and object formats
            if isinstance(chunk, dict):
                ticker = chunk["metadata"]["ticker"]
            else:
                ticker = chunk.metadata["ticker"]
            company_chunks[ticker].append(chunk)
        
        if not company_chunks: return []
        
        min_chunks = min(len(c) for c in company_chunks.values())
        print(f"🎯 Balancing to {min_chunks} chunks per company.")
        
        balanced = []
        for company, company_list in company_chunks.items():
            balanced.extend(random.sample(company_list, min_chunks))
            print(f"   - {company}: {min_chunks} chunks")
        
        return balanced

    def save_chunks_to_jsonl(self, chunks: List, path: Path):
        """Save chunk data to a JSONL file for inspection."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for chunk in chunks:
                # Handle both dict and object formats
                if isinstance(chunk, dict):
                    chunk_data = {
                        "id": chunk["id"],
                        "text": chunk["text"],
                        "metadata": chunk["metadata"],
                        "token_count": count_tokens(chunk["text"]),
                        "chunk_class": classify_chunk_by_tokens(chunk["text"])
                    }
                else:
                    chunk_data = {
                        "id": chunk.id,
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                        "token_count": count_tokens(chunk.text),
                        "chunk_class": classify_chunk_by_tokens(chunk.text)
                    }
                f.write(json.dumps(chunk_data) + "\n")

def generate_qa_pairs(chunks: List, 
                      output_path: str = Path(os.getcwd()) / 'qa_dataset.jsonl', 
                      append: bool = False,
                      debug_mode: bool = False):
    """
    Generate QA pairs using LangChain's ChatOpenAI.
    
    Args:
        chunks: List of chunk dictionaries or objects.
        output_path: Path to save the QA dataset.
        append: If True, append to existing file. Otherwise, overwrite.
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain.schema import HumanMessage, SystemMessage
        from tqdm.auto import tqdm
    except ImportError:
        print("❌ Required packages not installed. Run: pip install langchain langchain-openai tqdm")
        return

    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", max_tokens=300)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_mode = "a" if append else "w"
    
    system_prompt = """
You are a financial analyst assistant. Your job is to generate high-quality question-answer pairs based on SEC filing text.
INSTRUCTIONS:
1. Generate 2 specific, answerable questions based ONLY on the provided text.
2. Each question must explicitly include the company name and fiscal year.
3. Provide accurate, concise answers based solely on the text content.
4. Return your response as valid JSON in this exact format: {"qa_pairs": [{"question": "...", "answer": "..."}, ...]}
"""

    progress_bar = tqdm(chunks, desc="🤖 Generating QA pairs (via LangChain)", unit="chunk")
    with open(output_path, file_mode, encoding="utf-8") as f:
        for chunk in progress_bar:
            try:
                # Handle both dict and object formats
                if isinstance(chunk, dict):
                    chunk_text = chunk["text"]
                    chunk_id = chunk["id"]
                    metadata = chunk["metadata"]
                else:
                    chunk_text = chunk.text
                    chunk_id = chunk.id
                    metadata = chunk.metadata
                
                # Ensure metadata values are JSON serializable
                clean_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, (np.integer, np.int64)):
                        clean_metadata[key] = int(value)
                    elif isinstance(value, (np.floating, np.float64)):
                        clean_metadata[key] = float(value)
                    else:
                        clean_metadata[key] = value
                    
                user_prompt = f"Generate question-answer pairs for this SEC filing text:\n\n{chunk_text}"
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                
                response = llm.invoke(messages)
                time.sleep(1)  # Add rate limiting to avoid empty responses
                qa_data = json.loads(response.content.strip())

                for qa_pair in qa_data.get("qa_pairs", []):
                    question = qa_pair.get("question", "").strip()
                    answer = qa_pair.get("answer", "").strip()
                    
                    if question and answer:
                        # Handle both old ('item') and new ('section') field names
                        section = clean_metadata.get("section") or clean_metadata.get("item")
                        
                        qa_entry = {
                            "chunk_id": chunk_id,
                            "human_readable_id": clean_metadata.get("human_readable_id"),
                            "ticker": clean_metadata["ticker"],
                            "year": clean_metadata["fiscal_year"],
                            "section": section,
                            "section_num": clean_metadata.get("section_num"),
                            "section_letter": clean_metadata.get("section_letter"),
                            "question": question,
                            "answer": answer,
                            "source_text": chunk_text,
                        }
                        f.write(json.dumps(qa_entry) + "\n")
                        if debug_mode:
                            pprint(qa_entry)
                            print("-"*100)
            except Exception as e:
                progress_bar.write(f"⚠️ Error on chunk {chunk_id}: {e}")

# --- Main Orchestration ---

def main():
    """Main function to run the full QA generation pipeline."""
    # This script requires DocumentStore and SmartChunker, which might not be available
    # in the same environment as the langchain packages.
    try:
        from rag.document_store import DocumentStore
        from rag.chunkers import SmartChunker
    except ImportError:
        print("Could not import from rag package. Make sure it's installed and in the python path.")
        sys.exit(1)


    # Initialize the document store to get access to all sentences
    print("Initializing DocumentStore to get chunks...")
    doc_store = DocumentStore()
    df_sentences = doc_store.get_all_sentences()

    # Create section-level documents for chunking.
    # The SmartChunker expects sentence-level input, but for QA generation,
    # we want to chunk entire sections to get broader context.
    print("Aggregating sentences into section-level documents...")
    section_docs = df_sentences.groupby(['docID', 'ticker', 'fiscal_year', 'section']).agg(
        sentence=('sentence', ' '.join)
    ).reset_index()

    # The chunker expects a 'sentence_token_count' column.
    section_docs['sentence_token_count'] = section_docs['sentence'].apply(count_tokens)

    # Initialize the chunker
    chunker = SmartChunker(
        target_tokens=750,
        hard_ceiling=1000,
        overlap_tokens=150
    )
    all_chunks = chunker.run(section_docs)
    
    # Balance the chunks
    sampler = BalancedChunkSampler()
    grouped_chunks = sampler.group_chunks_by_keys(all_chunks)
    balanced_chunks = sampler.stratified_sample(grouped_chunks)

    # Generate QA pairs
    qa_output_path = QA_DATASET_PATH
    print(f"\n🚀 Generating QA pairs from {len(balanced_chunks)} balanced chunks...")
    generate_qa_pairs(balanced_chunks, str(qa_output_path))
    print(f"\n✅ All done! QA dataset saved to: {qa_output_path}")



#--------Helper Functions--------

def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def prepare_chunks_for_qa_generation(chunks: List[Any]) -> List[Dict[str, Any]]:
    """
    Convert chunk objects to dictionaries compatible with generate_qa_pairs.
    
    Args:
        chunks: List of chunk objects or dictionaries
        
    Returns:
        List of dictionaries with JSON-serializable values
    """
    prepared_chunks = []
    
    for chunk in chunks:
        if hasattr(chunk, 'to_dict'):
            # If chunk has a to_dict method, use it
            chunk_dict = chunk.to_dict()
        elif isinstance(chunk, dict):
            # Already a dictionary
            chunk_dict = chunk.copy()
        else:
            # Convert object attributes to dictionary
            chunk_dict = {
                "id": chunk.id,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "embedding": getattr(chunk, 'embedding', None)
            }
        
        # Ensure all values are JSON serializable
        chunk_dict = convert_numpy_types(chunk_dict)
        prepared_chunks.append(chunk_dict)
    
    return prepared_chunks

if __name__ == "__main__":
    sys.exit(main()) 