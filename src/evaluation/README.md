# SEC VectorStore Evaluation System

This directory contains tools for creating balanced evaluation datasets and generating QA pairs from SEC filing chunks.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r evaluation/requirements.txt
```

### 2. Set OpenAI API Key

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Generate QA Dataset

```bash
python evaluation/generate_qa_dataset.py
```

This will create:
- `data/balanced_chunks_for_eval.jsonl` - Balanced sample of chunks
- `data/qa_dataset.jsonl` - Generated QA pairs

## Overview

### Problem
- SEC filing chunks vary greatly in length and representation across companies
- Some companies (like META) have many more long chunks than others
- We need balanced evaluation data to avoid bias

### Solution
- **Stratified sampling** across key dimensions: company, year, section, chunk class
- **Balanced representation** ensuring each group contributes equally
- **LangChain QA generation** for high-quality question-answer pairs

## Components

### 1. Balanced Chunk Sampler (`generate_qa_dataset.py`)

```python
from evaluation.generate_qa_dataset import BalancedChunkSampler

sampler = BalancedChunkSampler(max_per_group=3)
grouped = sampler.group_chunks_by_keys(chunks)
balanced_chunks = sampler.stratified_sample(grouped)
```

**Features:**
- Groups chunks by `(ticker, fiscal_year, section, chunk_class)`
- Chunk classes: "short" (≤350 tokens), "medium" (≤500), "large" (>500)
- Limits chunks per group to prevent overrepresentation

### 2. QA Generation (`generate_qa_pairs()`)

```python
from evaluation.generate_qa_dataset import generate_qa_pairs

generate_qa_pairs(balanced_chunks, "data/qa_dataset.jsonl")
```

**Features:**
- Uses LangChain's `QAGenerationChain`
- Generates multiple QA pairs per chunk
- Handles errors gracefully
- Saves structured JSONL output

### 3. Standalone Script (`generate_qa_dataset.py`)

Complete pipeline that:
1. Loads chunks from `data/chunks.pkl`
2. Creates stratified sample
3. Generates QA pairs
4. Shows distribution statistics

## Output Format

### Balanced Chunks (`balanced_chunks_for_eval.jsonl`)
```json
{
  "id": "some_uuid",
  "text": "Risk factors include...",
  "metadata": {
    "ticker": "TSLA",
    "fiscal_year": 2020,
    "section": "1A",
    "section_num": "1",
    "section_letter": "A",
    "section_desc": "Risk Factors",
    "human_readable_id": "TSLA_2020_1A_0"
  },
  "token_count": 450,
  "chunk_class": "medium"
}
```

### QA Dataset (`qa_dataset.jsonl`)
```json
{
  "chunk_id": "TSLA_2020_1A_0",
  "ticker": "TSLA",
  "year": 2020,
  "section": "1A",
  "chunk_class": "medium",
  "question": "What are Tesla's main risk factors?",
  "answer": "Tesla faces risks including...",
  "source_text": "Full chunk text...",
  "token_count": 450
}
```

## Usage Examples

### In Notebook
```python
from evaluation.generate_qa_dataset import BalancedChunkSampler
import pickle

# Load chunks
with open("data/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Create balanced sample
sampler = BalancedChunkSampler(max_per_group=5)
grouped = sampler.group_chunks_by_keys(chunks)
balanced_chunks = sampler.stratified_sample(grouped)

print(f"Balanced sample: {len(balanced_chunks)} chunks")
```

### Command Line
```bash
# Generate full QA dataset
python evaluation/generate_qa_dataset.py

# Check the results
wc -l data/qa_dataset.jsonl
head -5 data/qa_dataset.jsonl
```

## Configuration

### Sampling Parameters
- `max_per_group`: Maximum chunks per `(ticker, year, section, chunk_class)` group
- Recommended: 3-5 for QA generation, 5-10 for larger evaluation sets

### Chunk Classification
- **Short**: ≤350 tokens (quick reads)
- **Medium**: 351-500 tokens (standard length)
- **Extra Large**: >500 tokens (comprehensive content)

### LangChain Settings
- Model: `gpt-4o-mini` (cost-efficient for QA generation)
- Temperature: 0 (consistent generation)

## Cost Estimation

For a balanced dataset with ~200 chunks:
- QA Generation: ~$2-5 (depending on chunk length)
- Embedding/Retrieval: ~$0.10-0.50

## Troubleshooting

### Common Issues

1. **LangChain Import Error**
   ```bash
   pip install langchain langchain-openai
   ```

2. **OpenAI API Key Missing**
   ```bash
   export OPENAI_API_KEY="your-key"
   ```

3. **chunks.pkl Not Found**
   - Ensure you've run the chunking pipeline
   - Check file path: `data/chunks.pkl`

4. **QA Generation Fails**
   - Check internet connection
   - Verify API key permissions
   - Monitor rate limits

### Debug Mode
```python
# Enable verbose output
generate_qa_pairs(chunks, output_path, verbose=True)
```

## Integration with Evaluation

Use the generated QA dataset with the SEC Query Evaluator:

```python
from src.evaluate import SECQueryEvaluator
import json

# Load QA pairs
qa_pairs = []
with open("data/qa_dataset.jsonl") as f:
    for line in f:
        qa_pairs.append(json.loads(line))

# Evaluate
evaluator = SECQueryEvaluator(use_docker=True)
for qa in qa_pairs[:5]:  # Test first 5
    result = evaluator.evaluate_query(qa["question"])
    print(f"Q: {qa['question']}")
    print(f"Expected: {qa['answer']}")
    print(f"Generated: {result['answer']}")
    print("---")
```

This creates a comprehensive evaluation framework for testing the SEC VectorStore system!

## Additional Details

### Balanced Chunk Sampling
The script includes a `BalancedChunkSampler` class with two key functions:
- `group_chunks_by_keys`: Groups chunks by `(ticker, fiscal_year, section, chunk_class)`. This ensures that documents are categorized based on their company, year, section, and complexity (length).
- `stratified_sample`: From these groups, it performs a stratified sample to create a balanced dataset. It also supports a `balance_companies` flag to ensure each company has the same number of chunks, preventing bias towards companies with more filings.

### QA Pair Generation
A `qa_entry` looks like this:
```json
{
  "chunk_id": "some_uuid",
  "human_readable_id": "TSLA_2020_1A_0",
  "ticker": "TSLA",
  "year": 2020,
  "section": "1A",
  "question": "What were the primary risk factors for Tesla in 2020?",
  "answer": "The primary risk factors for Tesla in 2020 included...",
  "source_text": "The full text of the chunk from which the QA pair was generated..."
}
```

The output is a JSONL file where each line is a dictionary containing a single QA pair.

Example `metadata` for a chunk:
```json
{
  "ticker": "TSLA",
  "fiscal_year": 2020,
  "section": "1A",
  "section_num": "1",
  "section_letter": "A",
  "section_desc": "Risk Factors",
  "human_readable_id": "TSLA_2020_1A_0"
}
```

## Usage

```bash
# Generate full QA dataset
python evaluation/generate_qa_dataset.py

# Check the results
wc -l data/qa_dataset.jsonl
head -5 data/qa_dataset.jsonl
```
