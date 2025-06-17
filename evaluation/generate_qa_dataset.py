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
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict

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
        """Group chunks by (ticker, fiscal_year, item, chunk_class)."""
        grouped = defaultdict(list)
        for chunk in chunks:
            # Handle both dict and object formats
            if isinstance(chunk, dict):
                metadata = chunk["metadata"]
                text = chunk["text"]
            else:
                metadata = chunk.metadata
                text = chunk.text
                
            key = (
                metadata["ticker"],
                metadata["fiscal_year"],
                metadata["item"],
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

def generate_qa_pairs(chunks: List, output_path: str):
    """Generate QA pairs using LangChain's ChatOpenAI."""
    try:
        from langchain_openai import ChatOpenAI
        from langchain.schema import HumanMessage, SystemMessage
        from tqdm.auto import tqdm
    except ImportError:
        print("❌ Required packages not installed. Run: pip install langchain langchain-openai tqdm")
        return

    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    system_prompt = """
You are a financial analyst assistant. Your job is to generate high-quality question-answer pairs based on SEC filing text.
INSTRUCTIONS:
1. Generate 2 specific, answerable questions based ONLY on the provided text.
2. Each question must explicitly include the company name and fiscal year.
3. Provide accurate, concise answers based solely on the text content.
4. Return your response as valid JSON in this exact format: {"qa_pairs": [{"question": "...", "answer": "..."}, ...]}
"""

    progress_bar = tqdm(chunks, desc="🤖 Generating QA pairs (via LangChain)", unit="chunk")
    with open(output_path, "w", encoding="utf-8") as f:
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
                    
                user_prompt = f"Generate question-answer pairs for this SEC filing text:\n\n{chunk_text}"
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                
                response = llm.invoke(messages)
                qa_data = json.loads(response.content.strip())

                for qa_pair in qa_data.get("qa_pairs", []):
                    question = qa_pair.get("question", "").strip()
                    answer = qa_pair.get("answer", "").strip()
                    
                    if question and answer:
                        qa_entry = {
                            "chunk_id": chunk_id,
                            "human_readable_id": metadata.get("human_readable_id"),
                            "ticker": metadata["ticker"],
                            "year": metadata["fiscal_year"],
                            "section": metadata["item"],
                            "question": question,
                            "answer": answer,
                            "source_text": chunk_text,
                        }
                        f.write(json.dumps(qa_entry) + "\n")
            except Exception as e:
                progress_bar.write(f"⚠️ Error on chunk {chunk_id}: {e}")

# --- Main Orchestration ---

def main():
    """Main function to generate the QA dataset."""
    print("🚀 Starting QA dataset generation...")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set.")
        return 1
    
    chunks_path = Path("data/chunks.pkl")
    if not chunks_path.exists():
        print(f"❌ {chunks_path} not found. Please create it first.")
        return 1
    
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    print(f"📊 Loaded {len(chunks)} chunks from {chunks_path}")
    
    print("\n🎯 Creating balanced sample...")
    sampler = BalancedChunkSampler(max_per_group=3, balance_companies=True)
    grouped = sampler.group_chunks_by_keys(chunks)
    balanced_chunks = sampler.stratified_sample(grouped)
    
    company_dist = Counter(c.metadata["ticker"] for c in balanced_chunks)
    print("\n📈 Final distribution after balancing:")
    print(f"   - By company: {dict(company_dist)}")
    
    balanced_chunks_path = Path("data/balanced_chunks_for_eval.jsonl")
    sampler.save_chunks_to_jsonl(balanced_chunks, balanced_chunks_path)
    print(f"\n💾 Balanced chunk sample saved for reference to {balanced_chunks_path}")
    
    print("\n🤖 Generating QA pairs...")
    qa_output_path = Path("data/qa_dataset.jsonl")
    generate_qa_pairs(balanced_chunks, str(qa_output_path))
    
    if qa_output_path.exists():
        with open(qa_output_path, "r") as f:
            lines = f.readlines()
        print(f"\n🎉 Success! Generated {len(lines)} QA pairs.")
        print(f"   -> Saved to: {qa_output_path}")
    else:
        print("\n❌ QA dataset generation failed.")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 