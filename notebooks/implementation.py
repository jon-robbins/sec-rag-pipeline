# %load_ext autoreload
# %autoreload 2

# %% [markdown]
#  # Implementation
# 
# 
# 
#  This notebook will walk you through the steps taken to implement the ensemble RAG's entire pipeline. For the baseline models you can see the implementation in `evaluation/scenarios.py`.

# %% [markdown]
#  ## Generate labeled data

# %% [markdown]
#  ## Data Preparation
# 
# 
# 
#  ### Data loading
# 
# 
# 
#  First we load the data. We'll use the `document_store.py` file for this.

# %%


import sys
from pathlib import Path

project_root = Path().absolute().parent
sys.path.insert(0, str(project_root))


from src.vector_store.document_store import DocumentStore

# Initialize the DocumentStore with default tickers
print("🔄 Initializing DocumentStore...")
raw_data_path = project_root / "data" / "raw" / "df_filings_full.parquet"
doc_store = DocumentStore(raw_data_path=raw_data_path)

# You can also specify custom tickers of interest:
# doc_store = DocumentStore(tickers_of_interest=['AAPL', 'META', 'GOOGL'])

# Load the full dataset
print("📁 Loading the full SEC filings dataset...")
full_dataset = doc_store.get_all_sentences()
full_dataset.head()


# %% [markdown]
#  ### Chunking
# 
# 
# 
#  We previously determined that the optimal chunking strategy is as follows:
# 
# 
# 
#  - 150 average tokens per chunk
# 
#  - 50 token overlap
# 
#  - 500 maximum token limit
# 
# 
# 
#  So we'll chunk the full dataset according to that.

# %%
from src.preprocessing.chunkers import Chunker, Chunk, ChunkingConfig

# Use the optimal chunking configuration determined from evaluation
chunking_config = ChunkingConfig(
    target_tokens=256,
    overlap_tokens=150, 
    hard_ceiling=1000
)

chunker = Chunker(config=chunking_config)

# Get properly formatted documents for chunking (fixes the design issue!)
print("🔄 Getting documents properly formatted for chunking...")
chunking_documents = doc_store.get_documents_for_chunking()

print(f"✅ Got {len(chunking_documents)} documents ready for chunking")
print(f"  Average tokens per document: {chunking_documents['total_tokens'].mean():.1f}")

# Now chunk these proper documents
chunk_df, stats = chunker.chunk_dataframe(chunking_documents)
chunks = [Chunk(**row) for row in chunk_df.to_dict('records')]  # Convert back to Chunk objects
print(f"✅ Created {len(chunks)} chunks using config: {chunking_config}")
print(f"Stats: {stats}")



# %% [markdown]
#  ### Retrieving embeddings
# 
# 
# 
#  We'll use OpenAI to get the embeddings for each chunk.

# %%
from src.vector_store.embedding import EmbeddingManager
import json
import os

# Set up logging to see detailed error messages
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Check if chunks already have embeddings
chunks_with_embeddings = [chunk for chunk in chunks if hasattr(chunk, 'embedding') and chunk.embedding is not None]

if len(chunks_with_embeddings) == len(chunks):
    print(f"✅ All {len(chunks)} chunks already have embeddings")
else:
    print(f"🔄 Generating embeddings for {len(chunks) - len(chunks_with_embeddings)} chunks...")
    embedding_manager = EmbeddingManager()
    
    # Only generate embeddings for chunks that don't have them
    chunks_needing_embeddings = [chunk for chunk in chunks if not hasattr(chunk, 'embedding') or chunk.embedding is None]
    texts = [chunk.text for chunk in chunks_needing_embeddings]
    
    # Debug: Check for problematic texts
    print(f"📊 Text analysis before embedding:")
    text_lengths = [len(text) for text in texts]
    print(f"  Min length: {min(text_lengths) if text_lengths else 0}")
    print(f"  Max length: {max(text_lengths) if text_lengths else 0}")
    print(f"  Average length: {sum(text_lengths)/len(text_lengths) if text_lengths else 0:.1f}")
    
    # Check for empty texts
    empty_texts = [i for i, text in enumerate(texts) if not text or not text.strip()]
    if empty_texts:
        print(f"⚠️ Found {len(empty_texts)} empty texts at indices: {empty_texts[:10]}")
    
    if texts:
        embeddings = embedding_manager.embed_texts_in_batches(texts)
        
        # Add embeddings to chunks that need them
        for i, chunk in enumerate(chunks_needing_embeddings):
            chunk.embedding = embeddings[i]
    
    print(f"✅ Generated embeddings for {len(chunks)} chunks total")

# Optionally save the chunks with embeddings for future use
save_dir = Path(os.getcwd()).parent / "data" / "implementation_example_files"
save_dir.mkdir(parents=True, exist_ok=True)

# Save as a simple JSON format that's compatible across refactoring
chunks_data = [
    {**chunk.model_dump(), "embedding": chunk.embedding}
    for chunk in chunks
]

with open(save_dir / "chunks_with_embeddings.json", "w") as f:
    json.dump(chunks_data, f, indent=2)

print(f"💾 Saved {len(chunks)} chunks with embeddings to chunks_with_embeddings.json")



# %% [markdown]
#  ## Generate labeled data
# 
# 
# 
#  We need to have ground truth to compare our RAG predictions to in order to evaluate their recall/precision. I will use `LangChain`'s OpenAI wrapper functionality to create QA pairs from chunks. There is some skepticism from the NLP community about the validity of LLM-generated training data or evaluation data, but due to resource/time limitations I'll assume that the LLM generated questions are valid. Considering the short context of the chunks given to the LLM, and the types of questions we're aiming for ("How much operating revenue did Tesla make in 2015?"), the risk that the metrics we obtain are entirely unreliable is low.
# 
# 
# 
#  In a real-world scenario, I would prefer to have a professionally labeled dataset with questions similar to what analysts/consultants may ask, with validated answers, along with daily quality checks of some sort, perhaps a rolling z-score deviation of the cosine similarity of certain clusters of documents, and an automated evaluation/tuning loop, but that's outside of the scope of this project.
# 
# 
# 
#  The following prompt is used:
# 
#  ```
# 
#  You are a financial analyst assistant. Your job is to generate high-quality question-answer pairs based on SEC filing text.
# 
#  INSTRUCTIONS:
# 
#  1. Generate 2 specific, answerable questions based ONLY on the provided text.
# 
#  2. Each question must explicitly include the company name and fiscal year.
# 
#  3. Provide accurate, concise answers based solely on the text content.
# 
#  4. Return your response as valid JSON in this exact format: {"qa_pairs": [{"question": "...", "answer": "..."}, ...]}
# 
#  ```

# %% [markdown]
#  But our first step is to stratify our sample queries to make sure that no company, year, or section is overrepresented in our evaluation set.

# %%

import random

from pathlib import Path

from src.openai_functions.qa_generation import (
    BalancedChunkSampler,
    generate_qa_pairs,
    prepare_chunks_for_qa_generation,
)

sampler = BalancedChunkSampler(max_per_group=5, min_tokens=50)
grouped_chunks = sampler.group_chunks_by_keys(chunks)
balanced_chunks = random.sample(
    sampler.stratified_sample(grouped_chunks), 300
)  

print(f"✅ Selected {len(balanced_chunks)} balanced chunks")


# %% [markdown]
#  Now we generate all the QA pairs.

# %%
import os
from pprint import pprint
qa_output_path = (
    Path(os.getcwd()).parent / "data" / "processed" / "qa_dataset_256798_tk.jsonl"
)
if Path.exists(qa_output_path):
    print(f"🎉 QA pairs already generated and saved to {qa_output_path}")
    prepared_chunks = [json.loads(line) for line in open(qa_output_path, "r")]
else:
    print(f"🔄 Generating QA pairs...")
    prepared_chunks = prepare_chunks_for_qa_generation(balanced_chunks[:10])
    generate_qa_pairs(prepared_chunks, qa_output_path, debug_mode=False)
    print(f"🎉 Generated ~{len(balanced_chunks)} questions saved to {qa_output_path}")


with open(qa_output_path, "r") as f:
    for line in f:
        data = json.loads(line)
        pprint(data)
        print("-" * 50)
    

# %% [markdown]
#  Notice that some of the questions don't specifically mention the company name, even when prompted. I played around with a lot of prompts to get it to generate the company name consistently, but to no avail. This could be the target for fine tuning at a later stage.
# 
# 
# 
#  My short term solution is to inject the information into the beginning of the question like so:
# 
# 
# 
#  ```python
# 
#  ```

# %% [markdown]
#  ## Parameter optimization

# %% [markdown]
#  First we should optimize the number of tokens per chunk split. I ran 50 questions on four different splits to optimize for recall, MRR, and Rouge.

# %%


import json
import os
import sys
from pathlib import Path
from pprint import pprint
import pandas as pd

sys.path.append(str(Path(os.getcwd()).parent))


data_path = Path(os.getcwd()).parent / "data"

from src.preprocessing.chunking_comparison import compare_chunking_configs

configs = [
    {"target_tokens": 150, "overlap_tokens": 25, "hard_ceiling": 500, "name": "Small_150_25"},
    # {"target_tokens": 300, "overlap_tokens": 50, "hard_ceiling": 800, "name": "Medium_300_50"},
    # {"target_tokens": 500, "overlap_tokens": 100, "hard_ceiling": 800, "name": "Large_500_100"},
    # {"target_tokens": 750, "overlap_tokens": 150, "hard_ceiling": 1000, "name": "XLarge_750_150"},
]

# Option to run fresh comparison or load cached results
run_fresh_comparison = True  # Set to True to run new comparison

if run_fresh_comparison:
    print("🔄 Running fresh chunking comparison...")
    df_results = compare_chunking_configs(num_questions=50, configs=configs)
    # Optionally save results
    # df_results.to_csv(data_path / 'chunking_comparison_results_new.csv')
else:
    # Load pre-computed results
    try:
        df_results = pd.read_csv(data_path / 'results' / 'archived_results'/'summaries'/'chunking_comparison_all_configs_20250620_184558.csv')
        print(f"✅ Loaded chunking comparison results: {df_results.shape}")
    except FileNotFoundError:
        print("⚠️ Chunking comparison results file not found.")
        print("Set run_fresh_comparison = True to generate new results.")
        df_results = pd.DataFrame()

display(df_results)


# %% [markdown]
# For simplicity, we'll look at Recall@5, RougeL, and nDCG@10
# 
# | Configuration          | Vanilla Recall\@5 | Reranked Recall\@5 | Ensemble Recall\@5 |
# | :--------------------- | :---------------: | :----------------: | :----------------: |
# | XLarge\_750\_150\_1000 |       0.040       |        0.060       |      **0.231**     |
# | Large\_500\_100\_800   |       0.160       |      **0.180**     |      **0.180**     |
# | Medium\_350\_100\_800  |       0.120       |      **0.140**     |      **0.140**     |
# | Small\_150\_50\_500    |       0.440       |      **0.540**     |        0.490       |
# 
# | Configuration          | Vanilla ROUGE-L | Reranked ROUGE-L | Ensemble ROUGE-L |
# | :--------------------- | :-------------: | :--------------: | :--------------: |
# | XLarge\_750\_150\_1000 |      0.101      |     **0.124**    |       0.122      |
# | Large\_500\_100\_800   |      0.323      |       0.355      |     **0.413**    |
# | Medium\_350\_100\_800  |      0.334      |       0.349      |     **0.424**    |
# | Small\_150\_50\_500    |      0.354      |       0.373      |     **0.428**    |
# 
# | Configuration          | Vanilla nDCG\@10 | Reranked nDCG\@10 | Ensemble nDCG\@10 |
# | :--------------------- | :--------------: | :---------------: | :---------------: |
# | XLarge\_750\_150\_1000 |       0.047      |       0.060       |     **0.202**     |
# | Large\_500\_100\_800   |       0.167      |     **0.180**     |       0.173       |
# | Medium\_350\_100\_800  |       0.127      |     **0.140**     |       0.133       |
# | Small\_150\_50\_500    |       0.388      |     **0.450**     |       0.413       |
# 
# 

# %% [markdown]
#  Takeaways:
# 
#  - Small configs consistently perform higher than other configs
# 
#  - Reranked in small configs perform better with recall and ndcg@10, but underperform with rouge. Meaning our reranker isn't reranking properly.
# 
# 
# 
#  Key takeaway for now is to keep the 150/50/500 batch size, and move on to testing all models.

# %% [markdown]
#  ## Baseline scenarios

# %% [markdown]
#  ### Vanilla `gpt-4o-mini`
# 
# 
# 
#  This implementation is simplest. We simply feed the API the question without context, and evaluate the answer.

# %%
# load qa set
with open(
    Path(os.getcwd()).parent / "data" / "processed" / "qa_dataset_256798_tk.jsonl", "r"
) as f:
    qa_set = [json.loads(line) for line in f]

from openai import OpenAI

from src.methods.baseline import run_baseline_scenario

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

qa_item = random.choice(qa_set)

baseline_output = run_baseline_scenario(openai_client, qa_item)
print(f"Question: {qa_item['question']}")
print(f"Expected: {qa_item['answer']}")
pprint(f"Baseline answer: {baseline_output[0]}")


# %% [markdown]
#  ### `gpt-4o-mini` with web search

# %%
from src.methods.web_search import run_web_search_scenario

response, tokens_used = run_web_search_scenario(openai_client, qa_item)

print(f"Question: {qa_item['question']}")
print(f"Expected: {qa_item['answer']}")
print(f"Web Search answer: {response}")

# %%


# %% [markdown]
#  ### `gpt-4o-mini` with full context
# 
# 
# 
#  This is the most wasteful but interesting baseline to use. It uploads an entire SEC 10-K filing as context, and gets the model to parse the whole document for the answer.

# %%
# Full context GPT search - shortest possible
from src.methods.unfiltered_text import run_unfiltered_context_scenario

# Load QA dataset and pick random question
with open(
    Path(os.getcwd()).parent / "data" / "processed" / "qa_dataset_300.jsonl", "r"
) as f:
    qa_set = [json.loads(line) for line in f]

qa_item = random.choice(qa_set)

# Run full context scenario (gets full filing text + asks question)
answer, token_usage = run_unfiltered_context_scenario(doc_store, openai_client, qa_item)

pprint(f"Question: {qa_item['question']}")
pprint(f"Expected: {qa_item['answer']}")
print(f"Full Context GPT: {answer}")
print(f"Tokens used: {token_usage['total_tokens']}")


# %% [markdown]
#  ## RAG scenarios

# %% [markdown]
#  ### Vanilla RAG

# %% [markdown]
#  This RAG will be very simple.
# 
# 
# 
#  ![Vanilla RAG](../images/vanilla-rag-flow.png)

# %% [markdown]
#  We send the embeddings into the vector DB.
# 
# 
# 
#  The user query is parsed through OpenAI to match their query to metadata if available. Specifically, extract a dictionary of `fiscal_year` and `ticker`. Only vectors that match that fiscal year and ticker are searched.
# 
# 
# 
#  The vector DB returns the top N vectors (currently N=10), which are then fed as context to Open AI to find the answer.

# %% [markdown]
#  #### Instantiate the RAG pipeline

# %% [markdown]
#  The `RAGPipeline` object will automatically call data; the above examples were for demonstration.

# %%
from src.vector_store.vector_store import VectorStore
from src.vector_store.embedding import EmbeddingManager

embedding_manager = EmbeddingManager()

# 0. Load chunks into vector DB with metadata and UUIDs
vs = VectorStore(use_docker=False, embedding_manager=embedding_manager)

# Prepare chunks with all metadata and IDs preserved
chunk_dicts = prepare_chunks_for_qa_generation(chunks)
embeddings_list = [chunk.embedding for chunk in chunks]

# Verify we have the right structure (metadata, id, text)
print(f"Sample chunk keys: {list(chunk_dicts[0].keys())}")
print(f"Sample chunk id: {chunk_dicts[0]['id']}")
print(f"Sample metadata: {chunk_dicts[0]['metadata']}")



# %% [markdown]
#  Now we upload the chunks into the vector store, ask a question,

# %%
# Upsert with embeddings, metadata, and UUIDs
vs.upsert_chunks(chunk_dicts, embeddings_list)
print(f"✅ Loaded {len(chunk_dicts)} chunks with metadata into vector DB")

# Use full RAG pipeline to retrieve and generate an answer
import json
import random

from src.openai_functions.answer_question import AnswerGenerator
from src.rag.pipeline import RAGPipeline

# 1️⃣ Load QA set and pick a random question
qa_set = [json.loads(line) for line in open(qa_output_path, "r")]
qa_item = random.choice(qa_set)

# 2️⃣ Build the pipeline
answer_generator = AnswerGenerator(openai_client)
rag_pipeline = RAGPipeline(vector_store=vs, answer_generator=answer_generator)

# 3️⃣ Retrieve relevant chunks & generate answer
search_results = rag_pipeline.search(qa_item["question"], top_k=10)
rag_response = rag_pipeline.generate_answer(qa_item["question"], search_results)

# 4️⃣ Display results
print(f"Question: {qa_item['question']}")
print(f"Expected: {qa_item['answer']}")
print(f"RAG answer: {rag_response['answer']}")
print(f"Retrieved {len(search_results)} chunks (top score {search_results[0]['score'] if search_results else 'N/A'})")


# %% [markdown]
#  ### RAG with Re-Ranker

# %% [markdown]
#  ![Reranking Rag](../images/rag-rerank-flow.png)

# %% [markdown]
#  With our re-ranker, we get the top 20 vectors by cosine similarity, and let the reranker get the ten most relevant vectors to send to the LLM.
# 
# 
# 
#  The BAAI/bge-reranker-base cross-encoder transformer assigns each query–vector pair a relevance logit. Unlike cosine similarity—which only measures the directional closeness of two independent embeddings, the reranker prepends/appends the query and document with [CLS] and [SEP] tokens, uses cross-attention to capture fine-grained semantic relations, and then ranks the vectors according to their logit scores.

# %%
from src.rag.reranker import BGEReranker
from src.openai_functions.answer_question import AnswerGenerator

reranker = BGEReranker()
answer_generator = AnswerGenerator(openai_client)
qa_item = random.choice(qa_set)

# Get 20 results, rerank to top 10, generate answer
query_embedding = embedding_manager.embed_texts_in_batches([qa_item["question"]])[0]
search_results = vs.search(query_vector=query_embedding, top_k=20)
texts = [r["payload"]["text"] for r in search_results]

reranked_indices = reranker.rerank(qa_item["question"], texts, top_k=10)
reranked_tuples = reranker.rerank(qa_item["question"], texts, top_k=10)

reranked_indices = [idx for idx, score in reranked_tuples]
reranked_results = [search_results[i] for i in reranked_indices]

result = answer_generator.generate_answer(qa_item["question"], reranked_results)



# %%
pprint(f"Question: {qa_item['question']}")
pprint(f"Expected: {qa_item['answer']}")
pprint(f"Reranked RAG: {result['answer']}")


# %% [markdown]
#  ### Ensemble Reranked RAG

# %% [markdown]
#  ![Ensemble RAG](../images/rag-ensemble-flow.png)

# %% [markdown]
#  After expanding the input query with an OpenAI call, the pipeline retrieves the top 20 documents by vector search and then applies two separate cross‐encoder rerankers, `BAAI/bge‐reranker‐base` and `jinaai/jina‐reranker‐v1‐base‐en` to each (query, document) pair. Each reranker outputs a relevance score with its [CLS]/[SEP] cross‐attention mechanism. Those scores are min–max normalized independently, averaged to form a fused score, and used to pick the final top 10. Finally, the selected passages are fed into a generative reader (AnswerGenerator) alongside the original question to produce the answer.

# %%
# Ensemble Reranked RAG - simplified version
from src.rag.reranker import BGEReranker
from sentence_transformers import CrossEncoder
import random
import numpy as np
# Initialize models
print("🔄 Loading ensemble rerankers...")
bge_reranker = BGEReranker()
jina_reranker = CrossEncoder(
    "jinaai/jina-reranker-v2-small-en", trust_remote_code=True
)
answer_generator = AnswerGenerator(openai_client)


qa_item = random.choice(qa_set)

# expand the query
expanded_query_prompt = f"""
Expand this financial question with relevant financial keywords and context:
Question: {qa_item['question']}

Return just the expanded question, nothing else.
"""

response = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": expanded_query_prompt}],
    max_tokens=100,
    temperature=0,
)
expanded_query = response.choices[0].message.content.strip()

# start retrieval
query_embedding = embedding_manager.embed_texts_in_batches([qa_item["question"]])[0]
search_results = vs.search(query_vector=query_embedding, top_k=20)
texts = [r["payload"]["text"] for r in search_results]

# ensemble reranking
bge_tuples = bge_reranker.rerank(expanded_query, texts, top_k=20)
bge_scores = np.array([score for idx, score in bge_tuples])

jina_scores = jina_reranker.predict([(expanded_query, text) for text in texts])

# normalize/fuse scores
bge_norm = (bge_scores - bge_scores.min()) / (
    bge_scores.max() - bge_scores.min() + 1e-6
)
jina_norm = (jina_scores - jina_scores.min()) / (
    jina_scores.max() - jina_scores.min() + 1e-6
)
fused_scores = (bge_norm + jina_norm) / 2

# get final results
final_indices = np.argsort(fused_scores)[::-1][:10]
final_results = [search_results[i] for i in final_indices]

# get the answer
result = answer_generator.generate_answer(qa_item["question"], final_results)

print(f"Original Query: {qa_item['question']}")
print(f"Expanded Query: {expanded_query}")
print(f"Expected: {qa_item['answer']}")
print(f"Ensemble RAG: {result['answer']}")


# %% [markdown]
#  # Next step: Run and evaluate
# 
# 
# 
#  See the notebook "Evaluation.ipynb" for comparing all models.



# %%
