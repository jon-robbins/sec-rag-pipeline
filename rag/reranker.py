import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class BGEReranker:
    """
    A reranker class using the BAAI/bge-reranker-base model.
    """
    def __init__(self, model_name: str = 'BAAI/bge-reranker-base'):
        self.device = "cuda" if torch.cuda.is_available() else \
                      "mps" if torch.backends.mps.is_available() else \
                      "cpu"
        print(f"BGEReranker using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def rerank(self, query: str, passages: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Reranks a list of passages based on a query.

        Args:
            query: The search query.
            passages: A list of passage texts to rerank.
            top_k: The number of top passages to return.

        Returns:
            A list of tuples, where each tuple contains the original index
            of the passage and its reranking score. The list is sorted by score.
        """
        if not passages:
            return []

        pairs = [[query, passage] for passage in passages]
        
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.device)
            scores = self.model(**inputs, return_dict=True).logits.view(-1).float()

        # Create a list of (original_index, score)
        indexed_scores = list(enumerate(scores.cpu().numpy()))
        
        # Sort by score in descending order
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        return indexed_scores[:top_k] 