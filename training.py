import numpy as np, torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          BitsAndBytesConfig, Trainer, TrainingArguments)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.nn.functional as F

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
USE_CLASS_WEIGHTS = False  # flip to True if your labels are imbalanced

# 1) Data
ds = load_dataset("ag_news")
train_val = ds["train"].train_test_split(test_size=0.1, seed=42, stratify_by_column="label")
train_ds, val_ds, test_ds = train_val["train"], train_val["test"], ds["test"]
num_labels = ds["train"].features["label"].num_classes  # 4

# 2) Tokenizer
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
tok.pad_token = tok.eos_token
tok.pad_token_id = tok.eos_token_id

def preprocess(ex):
    return tok(ex["text"], truncation=True, max_length=512)

def to_torch(dataset):
    d = dataset.map(preprocess, batched=True)
    d.set_format(type="torch", columns=["input_ids","attention_mask","label"])
    return d

train_t, val_t, test_t = to_torch(train_ds), to_torch(val_ds), to_torch(test_ds)

# 3) Model + LoRA (8-bit base)
quant = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.float16)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=num_labels, quantization_config=quant, device_map="auto"
)
lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.1, bias="none", task_type="SEQ_CLS",
    target_modules=["q_proj","k_proj","v_proj","o_proj"], modules_to_save=["score"]
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora)
model.config.pad_token_id = tok.pad_token_id

# 4) Optional class weights (only if imbalanced)
class_weights = None
if USE_CLASS_WEIGHTS:
    counts = np.bincount(train_t["label"].numpy(), minlength=num_labels)
    w = (counts.sum() / (num_labels * counts)).astype(np.float32)  # inverse-freq normalized
    class_weights = torch.tensor(w, dtype=torch.float32)

# 5) Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=1)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec}

# 6) Trainer (override loss only if using weights)
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if not USE_CLASS_WEIGHTS:
            return super().compute_loss(model, inputs, return_outputs)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = F.cross_entropy(outputs.logits, labels, weight=class_weights.to(outputs.logits.device))
        return (loss, outputs) if return_outputs else loss

args = TrainingArguments(
    output_dir="ag_news_lora",
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    num_train_epochs=1,               # bump to 3–5 for real runs
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
    fp16=True,
    gradient_checkpointing=True,
    warmup_ratio=0.1
)

trainer = WeightedTrainer(
    model=model, args=args,
    train_dataset=train_t, eval_dataset=val_t,
    tokenizer=tok, compute_metrics=compute_metrics
)

trainer.train()
print(trainer.evaluate(test_t))
