"""
Handles loading and generation of QA datasets for evaluation.
"""

import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Union

from ..rag.chunkers import Chunk
from .generate_qa_dataset import BalancedChunkSampler, generate_qa_pairs

logger = logging.getLogger(__name__)


class QADatasetManager:
    """
    Manages the loading and generation of QA datasets for evaluation.
    Creates config-specific datasets based on chunking parameters.
    """

    def __init__(self, root_dir: Path, chunking_config: Dict[str, int]):
        self.chunking_config = chunking_config

        # Create config-specific filename
        config_str = f"{chunking_config['target_tokens']}_{chunking_config['overlap_tokens']}_{chunking_config['hard_ceiling']}"
        self.config_tag = f"target_{config_str}"

        # Create processed data directory
        processed_dir = root_dir / "data" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Look for existing datasets with this configuration
        existing_datasets = list(
            processed_dir.glob(f"qa_dataset_{self.config_tag}_*.jsonl")
        )

        if existing_datasets:
            # Use the most recent existing dataset
            self.qa_dataset_path = max(
                existing_datasets, key=lambda p: p.stat().st_mtime
            )
            logging.info("Using existing QA dataset: %s", self.qa_dataset_path.name)
        else:
            # Create new dataset with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            qa_filename = f"qa_dataset_{self.config_tag}_{timestamp}.jsonl"
            self.qa_dataset_path = processed_dir / qa_filename
            logging.info("Will create new QA dataset: %s", qa_filename)

    def load_or_generate(
        self, num_questions: int, all_chunks: Union[List[Chunk], List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Loads the QA dataset, validates chunk compatibility, and generates more questions if needed.
        Only keeps QA pairs whose chunk IDs exist in the current pipeline.
        """
        # Create set of current pipeline chunk IDs for validation
        current_chunk_ids = set()
        if all_chunks:
            for chunk in all_chunks:
                if hasattr(chunk, "id"):
                    current_chunk_ids.add(chunk.id)
                elif isinstance(chunk, dict) and "id" in chunk:
                    current_chunk_ids.add(chunk["id"])

        qa_data = []
        if self.qa_dataset_path.exists():
            total_loaded = 0
            with open(self.qa_dataset_path, "r") as f:
                for line in f:
                    try:
                        qa_item = json.loads(line)
                        total_loaded += 1
                        # ✅ VALIDATE: Only keep QA pairs whose chunks exist in current pipeline
                        if qa_item.get("chunk_id") in current_chunk_ids:
                            qa_data.append(qa_item)
                        else:
                            logging.warning(
                                "Skipping QA pair - chunk %s not found in current pipeline",
                                qa_item.get("chunk_id"),
                            )
                    except json.JSONDecodeError:
                        logging.warning(
                            "Skipping malformed line in %s", self.qa_dataset_path
                        )

            logging.info(
                "Loaded %d QA pairs, %d compatible with current pipeline",
                total_loaded,
                len(qa_data),
            )
            logging.info("Pipeline has %d chunks", len(current_chunk_ids))
        else:
            logging.info("No existing QA dataset found at %s", self.qa_dataset_path)

        # Generate more questions if needed
        if len(qa_data) < num_questions:
            num_to_generate = num_questions - len(qa_data)
            self._generate_new_pairs(num_to_generate, all_chunks)
            # Reload and re-validate after generation
            qa_data = self._load_compatible_pairs(current_chunk_ids)

        # Final sampling
        if len(qa_data) >= num_questions:
            return random.sample(qa_data, num_questions)

        logging.warning(
            "Final dataset size (%d) is less than requested (%d). Using all available questions.",
            len(qa_data),
            num_questions,
        )
        return qa_data

    def _load_compatible_pairs(self, current_chunk_ids: set) -> List[Dict[str, Any]]:
        """Loads all QA pairs from the JSONL file and filters out those not in the current pipeline."""
        if not self.qa_dataset_path.exists():
            return []

        qa_data = []
        with open(self.qa_dataset_path, "r") as f:
            for line in f:
                try:
                    qa_item = json.loads(line)
                    if qa_item.get("chunk_id") in current_chunk_ids:
                        qa_data.append(qa_item)
                except json.JSONDecodeError:
                    continue  # Skip malformed lines silently in this helper method
        return qa_data

    def _generate_new_pairs(
        self, num_to_generate: int, all_chunks: Union[List[Chunk], List[Dict[str, Any]]]
    ):
        """Generates a specified number of new QA pairs."""
        logging.info("Generating %d missing questions...", num_to_generate)

        # Create a balanced sample of chunks for new QA generation
        max_per_group = max(3, min(10, (num_to_generate + 19) // 20))
        sampler = BalancedChunkSampler(max_per_group=max_per_group)
        grouped_chunks = sampler.group_chunks_by_keys(all_chunks)
        balanced_chunks = sampler.stratified_sample(grouped_chunks)

        logging.info(
            "Using max_per_group=%d for %d questions (found %d balanced chunks)",
            max_per_group,
            num_to_generate,
            len(balanced_chunks),
        )

        num_chunks_to_sample = (num_to_generate + 1) // 2
        if len(balanced_chunks) < num_chunks_to_sample:
            logging.warning(
                "Not enough balanced chunks (%d) to generate %d new questions. Using all available.",
                len(balanced_chunks),
                num_to_generate,
            )
            chunks_for_qa = balanced_chunks
        else:
            chunks_for_qa = random.sample(balanced_chunks, num_chunks_to_sample)

        logging.info(
            "Selected %d chunks to generate ~%d new questions.",
            len(chunks_for_qa),
            num_to_generate,
        )

        generate_qa_pairs(chunks_for_qa, self.qa_dataset_path, append=True)
