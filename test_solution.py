#!/usr/bin/env python3

import sys
sys.path.append('src')

from query_processor import AliasBuilder

# Test the improved system
print("=== Testing Improved Alias Resolution ===")

# Force refresh of embeddings to eliminate cache issues
builder = AliasBuilder(
    refresh_embeddings=True,  # This will rebuild embeddings
    score_cut=0.6  # Lower threshold for fuzzy matching
)

print("Building fresh alias index...")
client = builder.init_alias_index()

# Test the specific problematic queries
test_queries = [
    "What was meta's operating revenue in 2020?",
    "Facebok revenue 2017", 
    "Tesla revenues 2023",
    "How much profit did Microsoft make in 2019?"
]

print("\n=== Results ===")
for query in test_queries:
    ticker, score = builder.resolve_alias(client, query)
    status = "✅ RESOLVED" if ticker else "❌ NO MATCH"
    print(f"{status}: '{query}' → {ticker} (score={score:.3f})")

print("\nIf you see cache/embedding issues, delete the cache files and run again:")
print("rm -f embeddings/ticker_embeddings.pkl data/alias_table.parquet") 