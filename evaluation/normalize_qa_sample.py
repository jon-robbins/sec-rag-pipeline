import pandas as pd
import random
import json
import tiktoken
from collections import defaultdict, Counter
from typing import List, Dict
from pathlib import Path
from dataclasses import dataclass

# Use the actual Chunk class from the system
# We'll work with the chunk objects as they exist
ENCODER = tiktoken.encoding_for_model("gpt-4")

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    return len(ENCODER.encode(text))

def classify_chunk_by_tokens(text: str, ceiling: int = 800) -> str:
    """Classify chunk as extra_large or regular based on token count."""
    tokens = count_tokens(text)
    if tokens <= 350:
        return "short"
    elif tokens <= 500:
        return "medium"
    else:
        return "extra_large"

class BalancedChunkSampler:
    def __init__(self, max_per_group: int = 5, balance_companies: bool = True):
        self.max_per_group = max_per_group
        self.balance_companies = balance_companies

    def group_chunks_by_keys(self, chunks: List) -> Dict:
        """Group chunks by (ticker, fiscal_year, item, chunk_class)."""
        grouped = defaultdict(list)
        for chunk in chunks:
            # Work with the actual chunk structure
            key = (
                chunk.metadata["ticker"],
                chunk.metadata["fiscal_year"],
                chunk.metadata["item"],
                classify_chunk_by_tokens(chunk.text)
            )
            grouped[key].append(chunk)
        return grouped

    def stratified_sample(self, grouped_chunks: Dict) -> List:
        """Sample chunks evenly from each group with optional company balancing."""
        # First, sample from each group
        sampled = []
        for group, group_chunks in grouped_chunks.items():
            if len(group_chunks) > self.max_per_group:
                sampled.extend(random.sample(group_chunks, self.max_per_group))
            else:
                sampled.extend(group_chunks)
        
        # If company balancing is enabled, ensure equal representation
        if self.balance_companies:
            sampled = self._balance_by_company(sampled)
        
        return sampled
    
    def _balance_by_company(self, chunks: List) -> List:
        """Ensure equal representation across companies."""
        company_chunks = defaultdict(list)
        for chunk in chunks:
            company_chunks[chunk.metadata["ticker"]].append(chunk)
        
        # Find the minimum number of chunks across companies
        min_chunks = min(len(company_list) for company_list in company_chunks.values())
        print(f"🎯 Balancing to {min_chunks} chunks per company")
        
        balanced = []
        for company, company_chunk_list in company_chunks.items():
            if len(company_chunk_list) > min_chunks:
                selected = random.sample(company_chunk_list, min_chunks)
            else:
                selected = company_chunk_list
            balanced.extend(selected)
            print(f"   {company}: {len(selected)} chunks")
        
        return balanced

    def save_chunks_to_jsonl(self, chunks: List, path: Path):
        """Save chunks to JSONL format."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for chunk in chunks:
                record = {
                    "id": chunk.id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "token_count": count_tokens(chunk.text),
                    "chunk_class": classify_chunk_by_tokens(chunk.text)
                }
                f.write(json.dumps(record) + "\n")

def generate_qa_pairs(chunks: List, output_path: str):
    """Generate QA pairs using direct ChatOpenAI (more reliable than QAGenerationChain)."""
    try:
        from langchain_openai import ChatOpenAI
        from langchain.schema import HumanMessage, SystemMessage
    except ImportError:
        print("❌ LangChain not installed. Install with: pip install langchain langchain-openai")
        return

    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    successful_generations = 0
    failed_generations = 0
    total_qa_pairs = 0

    system_prompt = """You are a financial analyst assistant. Your job is to generate high-quality question-answer pairs based on SEC filing text.

INSTRUCTIONS:
1. Generate 2-3 specific, answerable questions based ONLY on the provided text
2. Each question should be clear and focused on factual information in the text
3. Provide accurate, concise answers based solely on the text content
4. Focus on: financial metrics, business operations, risk factors, competitive landscape, etc.
5. Return your response as valid JSON in this exact format:

{
  "qa_pairs": [
    {"question": "What was the revenue for Q1?", "answer": "Revenue was $X million"},
    {"question": "What are the main risk factors?", "answer": "The main risks include..."}
  ]
}"""

    with open(output_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            try:
                print(f"Generating QA for chunk {i+1}/{len(chunks)}: {chunk.id}")
                
                # Create the human message with chunk text
                user_prompt = f"Generate question-answer pairs for this SEC filing text:\n\n{chunk.text}"
                
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                
                # Get response from ChatOpenAI
                response = llm.invoke(messages)
                response_text = response.content.strip()
                
                # Parse JSON response
                try:
                    qa_data = json.loads(response_text)
                    qa_pairs = qa_data.get("qa_pairs", [])
                except json.JSONDecodeError:
                    print(f"⚠️ Invalid JSON response for chunk {chunk.id}, skipping...")
                    failed_generations += 1
                    continue
                
                # Write QA pairs
                for qa_pair in qa_pairs:
                    question = qa_pair.get("question", "").strip()
                    answer = qa_pair.get("answer", "").strip()
                    
                    if question and answer:
                        qa_entry = {
                            "chunk_id": chunk.id,
                            "ticker": chunk.metadata["ticker"],
                            "year": chunk.metadata["fiscal_year"],
                            "section": chunk.metadata["item"],
                            "chunk_class": classify_chunk_by_tokens(chunk.text),
                            "question": question,
                            "answer": answer,
                            "source_text": chunk.text,
                            "token_count": count_tokens(chunk.text)
                        }
                        f.write(json.dumps(qa_entry, ensure_ascii=False) + "\n")
                        total_qa_pairs += 1
                
                successful_generations += 1
                
            except Exception as e:
                print(f"⚠️ Failed to generate QA for chunk {chunk.id}: {e}")
                failed_generations += 1
                continue

    print(f"✅ QA generation complete!")
    print(f"   Successful chunks: {successful_generations}")
    print(f"   Failed chunks: {failed_generations}")
    print(f"   Total QA pairs: {total_qa_pairs}")
    print(f"   Saved to: {output_path}")

# Example usage
if __name__ == "__main__":
    import pickle
    from pathlib import Path

    # Load chunks
    try:
        with open("./data/chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        print(f"📊 Loaded {len(chunks)} chunks")
    except FileNotFoundError:
        print("❌ chunks.pkl not found. Make sure the file exists.")
        exit(1)

    # Create balanced sample
    sampler = BalancedChunkSampler(max_per_group=5, balance_companies=True)
    grouped = sampler.group_chunks_by_keys(chunks)
    balanced_chunks = sampler.stratified_sample(grouped)
    
    print(f"📊 Balanced sample: {len(balanced_chunks)} chunks from {len(grouped)} groups")
    
    # Show final distribution
    company_dist = Counter(chunk.metadata["ticker"] for chunk in balanced_chunks)
    chunk_class_dist = Counter(classify_chunk_by_tokens(chunk.text) for chunk in balanced_chunks)
    
    print(f"📈 Final distribution:")
    print(f"   By company: {dict(company_dist)}")
    print(f"   By chunk class: {dict(chunk_class_dist)}")
    
    # Save balanced chunks
    save_path = Path("./data/balanced_chunks_for_eval.jsonl")
    sampler.save_chunks_to_jsonl(balanced_chunks, save_path)
    print(f"✅ Saved balanced chunks to {save_path}")
    
    # Generate QA pairs
    qa_output_path = "./data/qa_dataset.jsonl"
    generate_qa_pairs(balanced_chunks, qa_output_path)
