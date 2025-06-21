"""
Handles loading and generation of QA datasets for evaluation.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Union

from ..rag.chunkers import Chunk
from .generate_qa_dataset import BalancedChunkSampler, generate_qa_pairs


class QADatasetManager:
    """
    Manages the loading and generation of QA datasets for evaluation.
    """

    def __init__(self, qa_dataset_path: Path, quiet: bool = False):
        self.qa_dataset_path = qa_dataset_path
        self.quiet = quiet

    def load_or_generate(
        self, num_questions: int, all_chunks: Union[List[Chunk], List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Loads the QA dataset, generates more questions if needed, and samples
        the requested number of questions for the evaluation.
        """
        qa_data = []
        if self.qa_dataset_path.exists():
            with open(self.qa_dataset_path, "r") as f:
                for line in f:
                    try:
                        qa_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        if not self.quiet:
                            print(
                                f"Warning: Skipping malformed line in {self.qa_dataset_path}"
                            )

        if not self.quiet:
            print(
                f"✅ Found {len(qa_data)} existing QA pairs in {self.qa_dataset_path}."
            )

        # Generate more questions if needed
        if len(qa_data) < num_questions:
            num_to_generate = num_questions - len(qa_data)
            self._generate_new_pairs(num_to_generate, all_chunks)
            # Reload after generation
            qa_data = self._load_from_file()

        # Final sampling
        if len(qa_data) >= num_questions:
            return random.sample(qa_data, num_questions)

        if not self.quiet:
            print(
                f"⚠️ Warning: Final dataset size ({len(qa_data)}) is less than requested ({num_questions}). Using all available questions."
            )
        return qa_data

    def _load_from_file(self) -> List[Dict[str, Any]]:
        """Loads all QA pairs from the JSONL file."""
        if not self.qa_dataset_path.exists():
            return []

        qa_data = []
        with open(self.qa_dataset_path, "r") as f:
            for line in f:
                try:
                    qa_data.append(json.loads(line))
                except json.JSONDecodeError:
                    if not self.quiet:
                        print(
                            f"Warning: Skipping malformed line in {self.qa_dataset_path}"
                        )
        return qa_data

    def _generate_new_pairs(
        self, num_to_generate: int, all_chunks: Union[List[Chunk], List[Dict[str, Any]]]
    ):
        """Generates a specified number of new QA pairs."""
        if not self.quiet:
            print(f"🧬 Generating {num_to_generate} missing questions...")

        # Create a balanced sample of chunks for new QA generation
        max_per_group = max(3, min(10, (num_to_generate + 19) // 20))
        sampler = BalancedChunkSampler(max_per_group=max_per_group)
        grouped_chunks = sampler.group_chunks_by_keys(all_chunks)
        balanced_chunks = sampler.stratified_sample(grouped_chunks)

        if not self.quiet:
            print(
                f"📊 Using max_per_group={max_per_group} for {num_to_generate} questions (found {len(balanced_chunks)} balanced chunks)"
            )

        num_chunks_to_sample = (num_to_generate + 1) // 2
        if len(balanced_chunks) < num_chunks_to_sample:
            if not self.quiet:
                print(
                    f"⚠️ Warning: Not enough balanced chunks ({len(balanced_chunks)}) to generate {num_to_generate} new questions. Using all available."
                )
            chunks_for_qa = balanced_chunks
        else:
            chunks_for_qa = random.sample(balanced_chunks, num_chunks_to_sample)

        if not self.quiet:
            print(
                f"Selected {len(chunks_for_qa)} chunks to generate ~{num_to_generate} new questions."
            )

        generate_qa_pairs(chunks_for_qa, self.qa_dataset_path, append=True)
