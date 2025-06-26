from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import Dataset, DatasetDict
from typing import List, Tuple
import numpy as np
import os


class NERTrainer:
    def __init__(self, model_name: str, label_list: List[str]):
        self.model_name = model_name
        self.label_list = label_list
        self.label_to_id = {l: i for i, l in enumerate(label_list)}
        self.id_to_label = {i: l for i, l in enumerate(label_list)}

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(label_list),
            id2label=self.id_to_label,
            label2id=self.label_to_id
        )


    def load_conll_data(self, filepath: str, split: float = 0.2, seed: int = 42) -> DatasetDict:
        """
        Parse CoNLL format and return a train/validation split.
        """
        data = []
        tokens = []
        labels = []

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    if tokens:
                        data.append({"tokens": tokens, "ner_tags": [self.label_to_id[t] for t in labels]})
                        tokens, labels = [], []
                else:
                    splits = line.split()
                    if len(splits) == 2:
                        token, label = splits
                        tokens.append(token)
                        labels.append(label)

        dataset = Dataset.from_list(data)
        dataset_split = dataset.train_test_split(test_size=split, seed=seed)
        return dataset_split



    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True
        )

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            aligned_labels = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    aligned_labels.append(-100)
                elif word_idx != previous_word_idx:
                    aligned_labels.append(label[word_idx])
                else:
                    aligned_labels.append(label[word_idx] if self.label_list[label[word_idx]].startswith("I-") else -100)
                previous_word_idx = word_idx
            labels.append(aligned_labels)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs


    def train(self, train_dataset: Dataset, val_dataset: Dataset, output_dir="ner_model"):
        tokenized_train = train_dataset.map(self.tokenize_and_align_labels, batched=True)
        tokenized_val = val_dataset.map(self.tokenize_and_align_labels, batched=True)

        args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            save_strategy="epoch",
            logging_dir=f"{output_dir}/logs",
        )

        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )

        trainer.train()
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
