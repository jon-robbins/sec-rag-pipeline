import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from tqdm import tqdm
from openai import OpenAI          # → `OPENAI_API_KEY` env var
import pickle
client = OpenAI()

def embed_texts(chunks, model="text-embedding-3-small", batch=256):
    ids, vectors, metas = [], [], []
    for i in range(0, len(chunks), batch):
        batch_chunks = chunks[i:i+batch]
        resp = client.embeddings.create(
            model=model,
            input=[c.text for c in batch_chunks]
        )
        vectors.extend([d.embedding for d in resp.data])
        ids.extend([c.id for c in batch_chunks])
        metas.extend([c.metadata for c in batch_chunks])
    return ids, vectors, metas


