file = '/Users/jon/GitHub/dowjones-takehome/data/processed/qa_dataset.jsonl'
import json
from tiktoken import encoding_for_model

# Initialize tokenizer
enc = encoding_for_model("gpt-4")
max_tokens = 0

with open(file, 'r') as f:
    for line in f:
        data = json.loads(line)
        tokens = len(enc.encode(data['answer']))
        max_tokens = max(max_tokens, tokens)

print(f"Maximum number of tokens in any answer: {max_tokens}")
