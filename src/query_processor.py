# alias_builder.py – v3.2
"""
Provides AliasBuilder, a class-based interface for building and querying
an alias-to-ticker vector index with persistent embedding cache, using the
shared embed_texts utility (src/embedding.py) for all embedding operations.

Usage:
    builder = AliasBuilder(
        parquet_path="data/alias_table.parquet",
        cache_path="embeddings/ticker_embeddings.pkl",
        refresh_table=True,
        refresh_embeddings=False,
        embedding_model="text-embedding-3-small"
    )
    df = builder.load_table()
    client = builder.init_alias_index(existing_client)
    ticker, score = builder.resolve_alias(client, "Facebok revenue 2017")
    builder.sanity_demo()

Requirements:
    - pandas, requests, openai, qdrant-client, tiktoken
    - embed_texts in src/embedding.py
    - OPENAI_API_KEY in env
"""
from __future__ import annotations
import os, re, time, pickle
from types import SimpleNamespace
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import requests
import openai
from qdrant_client import QdrantClient, models
from embedding import embed_texts

class AliasBuilder:
    def __init__(
        self,
        parquet_path: str | Path = "data/alias_table.parquet",
        cache_path: str | Path = "embeddings/ticker_embeddings.pkl",
        refresh_table: bool = True,
        refresh_embeddings: bool = False,
        embedding_model: str = "text-embedding-3-small",
        alias_collection: str = "alias_index",
        vector_size: int = 1536,
        score_cut: float = 0.6,  # Default lower threshold for fuzzy matching
        openai_key: Optional[str] = None,
    ):
        self.parquet_path = Path(parquet_path)
        self.cache_path = Path(cache_path)
        self.refresh_table = refresh_table
        self.refresh_embeddings = refresh_embeddings
        self.embedding_model = embedding_model
        self.alias_collection = alias_collection
        self.vector_size = vector_size
        self.default_score_cut = score_cut
        openai.api_key = openai_key or os.getenv("OPENAI_API_KEY", "")
        self._df: Optional[pd.DataFrame] = None

    def _fetch_sec(self, max_retry: int = 3, backoff: float = 1.5) -> List[dict]:
        url = "https://www.sec.gov/files/company_tickers.json"
        headers = {"User-Agent": "alias-builder/2.1 (contact: jrrobbins@gmail.com)"}
        for attempt in range(max_retry):
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.ok:
                return list(resp.json().values())
            if attempt == max_retry - 1:
                resp.raise_for_status()
            time.sleep(backoff * (2 ** attempt))
        return []

    def load_table(self) -> pd.DataFrame:
        """Load or build the alias DataFrame with embeddings."""
        if self._df is not None:
            return self._df
        # Build or load table rows
        if self.parquet_path.exists() and not self.refresh_table:
            df = pd.read_parquet(self.parquet_path)
        else:
            rows = []
            for rec in self._fetch_sec():
                ticker = rec['ticker'].upper()
                # ticker symbol alias
                rows.append({
                    'ticker': ticker,
                    'alias': ticker.lower(),
                    'from_year': 2000,
                    'to_year': 9999,
                    'cik': int(rec['cik_str']),
                })
                # normalized full title alias (with spaces)
                title_norm = re.sub(r"[^a-z0-9 ]", "", rec['title'].lower())
                rows.append({
                    'ticker': ticker,
                    'alias': title_norm,
                    'from_year': 2000,
                    'to_year': 9999,
                    'cik': int(rec['cik_str']),
                })
                # n-gram aliases (1-4 tokens) with spaces
                tokens = title_norm.split()
                max_n = min(4, len(tokens))
                for n in range(1, max_n + 1):
                    for i in range(len(tokens) - n + 1):
                        alias_ng = ' '.join(tokens[i:i+n])
                        rows.append({
                            'ticker': ticker,
                            'alias': alias_ng,
                            'from_year': 2000,
                            'to_year': 9999,
                            'cik': int(rec['cik_str']),
                        })
            df = pd.DataFrame(rows)
        # Compute or load embeddings
        if self.cache_path.exists() and not self.refresh_embeddings:
            with open(self.cache_path, 'rb') as f:
                ids, vecs, metas = pickle.load(f)
        else:
            chunks = [SimpleNamespace(id=str(i), text=a, metadata={})
                      for i, a in enumerate(df['alias'])]
            ids, vecs, metas = embed_texts(
                chunks,
                model=self.embedding_model,
                refresh=self.refresh_embeddings
            )
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, 'wb') as f:
                pickle.dump((ids, vecs, metas), f)
        if len(vecs) != len(df):
            raise ValueError(f"Embeddings length {len(vecs)} != {len(df)}")
        df['embedding'] = vecs
        df.to_parquet(self.parquet_path, compression='zstd')
        self._df = df
        return df

    def init_alias_index(self, client: Optional[QdrantClient] = None) -> QdrantClient:
        """Create or reuse a QdrantClient and upsert alias embeddings."""
        df = self.load_table()
        if client is None:
            client = QdrantClient(":memory:")
        client.recreate_collection(
            collection_name=self.alias_collection,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        points = []
        for i, row in df.iterrows():
            points.append(models.PointStruct(
                id=i,
                vector=row['embedding'],
                payload={'ticker': row['ticker'], 'alias': row['alias']}
            ))
        client.upsert(collection_name=self.alias_collection, points=points)
        return client

    def _extract_company_candidates(self, query: str) -> List[str]:
        """Extract multiple candidate company phrases from query using various strategies."""
        candidates = []
        
        # Strategy 1: spaCy NER for ORG entities
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
            doc = nlp(query)
            orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
            candidates.extend(orgs)
        except Exception:
            pass
        
        # Strategy 2: Noun chunks that might be company names
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(query)
            question_words = {'what', 'who', 'when', 'where', 'why', 'how', 'which'}
            for chunk in doc.noun_chunks:
                chunk_text = chunk.text.strip()
                if len(chunk_text) > 2 and chunk_text.lower() not in question_words:
                    candidates.append(chunk_text)
        except Exception:
            pass
        
        # Strategy 3: Capitalized words/phrases (original fallback), filtering stopwords
        cap_matches = re.findall(r"([A-Z][A-Za-z0-9''&. ]{1,})", query)
        # Filter out question words and common stopwords
        question_stopwords = {
            'what', 'was', 'how', 'much', 'did', 'make', 'in', 'the', 'and', 'or', 'but', 
            'revenue', 'profit', 'income', 'risks', 'factors', 'operating', 'gross', 
            'margin', 'net', 'faced', 'year', 'years', 'financial', 'report', 'for',
            'who', 'when', 'where', 'why', 'which', 'would', 'could', 'should'
        }
        for match in cap_matches:
            if match.split()[0].lower() not in question_stopwords:  # Check first word
                candidates.append(match)
        
        # Strategy 4: Possessive forms (e.g., "Tesla's", "Meta's")
        possessive_matches = re.findall(r"([A-Za-z][A-Za-z0-9''&.]{2,})'s", query, re.IGNORECASE)
        candidates.extend(possessive_matches)
        
        # Strategy 5: All potential company words (filtering common stopwords)
        stopwords = {
            'what', 'was', 'how', 'much', 'did', 'make', 'in', 'the', 'and', 'or', 'but', 
            'revenue', 'profit', 'income', 'risks', 'factors', 'operating', 'gross', 
            'margin', 'net', 'faced', 'year', 'years', 'financial', 'report', 'for',
            'who', 'when', 'where', 'why', 'which', 'would', 'could', 'should'
        }
        words = re.findall(r"[A-Za-z][A-Za-z0-9''&.]{2,}", query)
        for word in words:
            if word.lower() not in stopwords:
                candidates.append(word)
        
        # Strategy 6: First alphabetic sequence (debug_resolve approach)
        m = re.search(r"([A-Za-z][A-Za-z0-9''&. ]{2,})", query)
        if m:
            candidates.append(m.group(1))
        
        # Remove duplicates and empty strings
        candidates = list(set([c.strip() for c in candidates if c.strip()]))
        
        # Smart prioritization: prefer single words/short phrases over long ones
        def candidate_priority(phrase):
            # Remove the full query from consideration (too generic)
            if phrase.strip().lower() == query.strip().lower():
                return 1000  # lowest priority
            
            # Count number of words
            words = phrase.split()
            word_count = len(words)
            
            # Check if it starts with a capital letter (likely a proper noun)
            is_capitalized = phrase and phrase[0].isupper()
            
            # Prioritize 1-3 word phrases, penalize longer ones
            # Within each category, prioritize capitalized words
            if word_count == 1:
                return 0.5 if is_capitalized else 1  # capitalized single words highest priority
            elif word_count == 2:
                return 1.5 if is_capitalized else 2
            elif word_count == 3:
                return 2.5 if is_capitalized else 3
            else:
                base_score = 10 + word_count
                return base_score - 0.5 if is_capitalized else base_score
        
        # Sort by priority (lower number = higher priority)
        candidates.sort(key=candidate_priority)
        
        # If no good candidates found, use whole query as fallback
        if not candidates or all(c.strip().lower() == query.strip().lower() for c in candidates):
            candidates = [query]
        
        return candidates

    def resolve_alias(
        self,
        client: QdrantClient,
        query: str,
        top_k: int = 1,
        score_cut: Optional[float] = None,  # Use default if not provided
    ) -> Tuple[Optional[str], float]:
        """Find best company match by testing multiple candidate phrases."""
        if score_cut is None:
            score_cut = self.default_score_cut
            
        candidates = self._extract_company_candidates(query)
        
        best_ticker = None
        best_score = 0.0
        best_phrase = None
        
        # Test each candidate phrase
        for phrase in candidates:
            # Consistent normalization (keep spaces like alias index)
            normed = re.sub(r"[^a-z0-9 ]", "", phrase.lower()).strip()
            if len(normed) < 2:
                continue
                
            chunk = SimpleNamespace(id='q', text=normed, metadata={})
            _, vecs, _ = embed_texts([chunk], model=self.embedding_model, refresh=False)
            qvec = vecs[0]
            
            hits = client.search(
                collection_name=self.alias_collection,
                query_vector=qvec,
                limit=1,
            )
            
            if hits and hits[0].score > best_score:
                best_score = hits[0].score
                best_ticker = hits[0].payload.get('ticker')
                best_phrase = phrase
        
        # Debug output for troubleshooting
        if best_ticker:
            print(f"DEBUG: '{query}' → extracted '{best_phrase}' → {best_ticker} (score={best_score:.3f})")
        else:
            print(f"DEBUG: '{query}' → no match above threshold {score_cut} (best score={best_score:.3f})")
        
        if best_score < score_cut:
            return None, 0.0
        
        return best_ticker, best_score

    def debug_resolve(
        self,
        client: QdrantClient,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[str, str, float]]:
        """Return the top-K alias hits for a query: (alias, ticker, score)."""
        # Use the same candidate extraction as resolve_alias for consistency
        candidates = self._extract_company_candidates(query)
        
        if not candidates:
            return []
        
        # Use the first (longest) candidate for debug
        phrase = candidates[0]
        normed = re.sub(r"[^a-z0-9 ]", "", phrase.lower()).strip()
        chunk = SimpleNamespace(id='q', text=normed, metadata={})
        _, vecs, _ = embed_texts([chunk], model=self.embedding_model, refresh=False)
        qvec = vecs[0]
        
        # Search alias index
        hits = client.search(
            collection_name=self.alias_collection,
            query_vector=qvec,
            limit=top_k,
        )
        
        print(f"DEBUG_RESOLVE: '{query}' → extracted '{phrase}' → normed '{normed}'")
        
        # Return list of (alias, ticker, score)
        return [(hit.payload.get('alias'), hit.payload.get('ticker'), hit.score) for hit in hits]

    def sanity_demo(
        self,
        queries: Optional[List[str]] = None,
        top_k: int = 1,
        score_cut: Optional[float] = None,  # Use default if not provided
    ) -> None:
        """Run a set of queries and print resolution results."""
        if score_cut is None:
            score_cut = self.default_score_cut
            
        if queries is None:
            queries = [
                "What was meta's operating revenue in 2020?",
                "Facebok revenue 2017",
                "How much profit did Microsof make in 2019?",
                "Risks Teslla faced in 2018",
                "Operting income for Amazn 2016",
                "Revenue of Alphbet 2015",
                "What's Lululemn's gross margin 2021",
                "Net income of Appl 2014",
                "Rsk factors for Wlamart 2022",
                "Tesla revenues 2023",
            ]
        client = self.init_alias_index()
        for q in queries:
            ticker, score = self.resolve_alias(client, q, top_k, score_cut)
            cleaned = q
            if ticker:
                # Replace the best matched phrase with the ticker
                candidates = self._extract_company_candidates(q)
                if candidates:
                    # Find which candidate was the best match by testing again
                    best_phrase = None
                    best_test_score = 0.0
                    for phrase in candidates:
                        normed = re.sub(r"[^a-z0-9 ]", "", phrase.lower()).strip()
                        if len(normed) < 2:
                            continue
                        chunk = SimpleNamespace(id='q', text=normed, metadata={})
                        _, vecs, _ = embed_texts([chunk], model=self.embedding_model, refresh=False)
                        qvec = vecs[0]
                        hits = client.search(
                            collection_name=self.alias_collection,
                            query_vector=qvec,
                            limit=1,
                        )
                        if hits and hits[0].score > best_test_score:
                            best_test_score = hits[0].score
                            best_phrase = phrase
                    
                    if best_phrase:
                        cleaned = q.replace(best_phrase, ticker, 1)
            
            print(
                f"ORIG: {q}\n"
                f" →  ticker={ticker or 'NO HIT'} (score={score:.3f})\n"
                f" ⇒  {cleaned}\n"
            )

if __name__ == "__main__":
    builder = AliasBuilder(
        parquet_path="data/alias_table.parquet",
        cache_path="embeddings/ticker_embeddings.pkl",
        refresh_table=False,
        refresh_embeddings=True,
        embedding_model="text-embedding-3-small",
        alias_collection="alias_index",
        vector_size=1536,
    )
    builder.sanity_demo()
