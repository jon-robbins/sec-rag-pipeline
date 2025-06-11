######################## MONKEY PATCHING########################
# monkey_patch_accelerate.py
from accelerate import Accelerator

# Keep a reference to the original
_original_unwrap = Accelerator.unwrap_model

def unwrap_model_no_compile(self, model, *args, **kwargs):
    # Remove the problematic kwarg if present
    kwargs.pop("keep_torch_compile", None)
    return _original_unwrap(self, model, *args, **kwargs)

# Apply the patch
Accelerator.unwrap_model = unwrap_model_no_compile

# monkey_patch_optimizer.py
import torch.optim

# Alias the .train() call to .step() for AdamW
if not hasattr(torch.optim.AdamW, "train"):
    torch.optim.AdamW.train = torch.optim.AdamW.step
########################END MONKEY PATCHING########################

from typing import List
import re
from pathlib import Path

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments, Trainer
)

class FinERTuner:
    def __init__(
        self,
        model_id: str = "nlpaueb/sec-bert-shape",
        dataset_id: str = "nlpaueb/finer-139",
        train_size: int = 100,
        eval_size:  int = 20,
        seed:       int = 42
    ):
        self.tokenizer   = AutoTokenizer.from_pretrained(model_id)
        raw_ds           = load_dataset(dataset_id)
        self.label_list  = raw_ds["train"].features["ner_tags"].feature.names
        self.model       = AutoModelForTokenClassification.from_pretrained(
                              model_id, num_labels=len(self.label_list)
                           )
        # SUBSET BEFORE TOKENIZATION
        self.dataset = DatasetDict({
            "train":      raw_ds["train"].shuffle(seed=seed).select(range(train_size)),
            "validation": raw_ds["validation"].shuffle(seed=seed).select(range(eval_size))
        })
        self.collator   = DataCollatorForTokenClassification(self.tokenizer)

    @staticmethod
    def shape_pseudo_token(text: str) -> str:
        def shape(num: str):
            groups = num.replace(",", "")
            parts  = groups.split(".")
            left   = len(parts[0])
            right  = len(parts[1]) if len(parts) > 1 else 0
            return f"[{'X'*left}{'.'+'X'*right if right else ''}]"

        return re.sub(r"\d[\d,]*\.?\d*", lambda m: shape(m.group()), text)

    def tokenize_and_align_labels(self, example):
        # Apply numeric-shape substitution
        text      = self.shape_pseudo_token(" ".join(example["tokens"]))
        tokenized = self.tokenizer(
            text.split(),
            is_split_into_words=True,
            truncation=True
        )
        word_ids  = tokenized.word_ids()
        new_labels = []
        prev_widx  = None

        for widx in word_ids:
            if widx is None:
                new_labels.append(-100)
            elif widx != prev_widx:
                new_labels.append(example["ner_tags"][widx])
            else:
                tag = example["ner_tags"][widx]
                # if continuation token, use I- label logic
                if self.label_list[tag].startswith("I"):
                    new_labels.append(tag)
                else:
                    # shift to I- tag by offsetting into second half
                    new_labels.append(tag + len(self.label_list)//2)
            prev_widx = widx

        tokenized["labels"] = new_labels
        return tokenized

    def prepare(self):
        # Tokenize only the small subsets
        self.tokenized_ds = self.dataset.map(
            self.tokenize_and_align_labels,
            batched=False,
            remove_columns=["tokens", "ner_tags"],
            load_from_cache_file=True  # uses cache if present
        )

    def get_trainer(self, output_dir: Path = Path("models") / "finer_shaped"):
        output_dir.mkdir(parents=True, exist_ok=True)
        args = TrainingArguments(
            output_dir=str(output_dir),
            eval_strategy="steps",       # or eval_strategy="epoch"
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            save_steps=500,
            eval_steps=500,
            logging_steps=100,
            weight_decay=0.01,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
        )
        return Trainer(
            model=self.model,
            args=args,
            train_dataset=self.tokenized_ds["train"],
            eval_dataset=self.tokenized_ds["validation"],
            processing_class=self.tokenizer,  # updated API
            data_collator=self.collator,
        )


if __name__ == "__main__":
    tuner = FinERTuner(train_size=100, eval_size=20)
    tuner.prepare()
    trainer = tuner.get_trainer()
    trainer.train()
    trainer.evaluate()
