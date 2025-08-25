# %%
# If running in a notebook, install required packages with:
# !pip install -q bitsandbytes accelerate transformers datasets peft evaluate scikit-learn

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# %%
# Load the AG News dataset


def load_ag_news():
    dataset = load_dataset("ag_news")
    split = dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    val_ds = split["test"]
    test_ds = dataset["test"]
    return train_ds, val_ds, test_ds


train_ds, val_ds, test_ds = load_ag_news()

# %%
# Model and tokenizer setup


model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id


def preprocess(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)


def tokenize_dataset(dataset):
    tokenized = dataset.map(preprocess, batched=True)
    token_columns = ["input_ids", "attention_mask", "label"]
    tokenized.set_format("torch", columns=token_columns)
    return tokenized


tokenized_train = tokenize_dataset(train_ds)
tokenized_val = tokenize_dataset(val_ds)
tokenized_test = tokenize_dataset(test_ds)

# %%
# Model loading with LoRA configuration


quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_use_double_quant=False,
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=4,
    quantization_config=quant_config,
    device_map="auto",
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.config.pad_token_id = tokenizer.pad_token_id

# %%
# Metrics and training setup


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


training_args = TrainingArguments(
    output_dir="ag_news_lora",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
    fp16=True,
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
results = trainer.evaluate(tokenized_test)
print(results)
