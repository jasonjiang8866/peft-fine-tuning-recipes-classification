#! pip install bitsandbytes accelerate transformers datasets bertviz polars peft tqdm evaluate scikit-learn py7zr --quiet
#! huggingface-cli login
'''
in case need to unzip files
import py7zr
with py7zr.SevenZipFile("Meta-Llama-3-8B-Instruct.7z", mode='r') as z:
    z.extractall(path="./project")
'''
# How to Finetune Llama 3 for Sequence Classification
import polars as pl
import os
os.environ['HF_HOME'] = '/project_cache'
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification,BitsAndBytesConfig
from datasets import load_dataset
import torch
import random
from tqdm import tqdm
from transformers import DataCollatorWithPadding
import numpy as np
from sklearn.metrics import classification_report, balanced_accuracy_score, accuracy_score, precision_score, average_precision_score
from transformers import Trainer, TrainingArguments
import torch.nn.functional as F

#read data
df = pl.read_parquet("data.parquet")
#encoding labels
text_labels = []
label_map = {label:i for i, label in enumerate(text_labels)}
df = df.with_columns(pl.col("label").replace_strict(label_map).alias("labels")).drop("label")
#resample data to balance the sample
df_balanced = df.group_by('labels').agg(pl.col('signal').shuffle().head(sample_n)).sort("labels").explode("signal")
df_balanced.write_parquet("balanced_data.parquet")

#sampling fn
def sample_ds(ds, sample_size):
    index_full = list(range(len(ds)))
    sample_index = random.sample(index_full, sample_size)
    res = ds.select(sample_index)
    return res

#tokenizer fn
def data_preprocesing(row):
    return tokenizer(row['signal'], truncation=True, max_length=1024)
  
# for multiclass
def compute_metrics_multiclass(evaluations):
    predictions, labels = evaluations
    predictions = np.argmax(predictions, axis=1)
    return {'balanced_accuracy' : balanced_accuracy_score(predictions, labels),
    'accuracy':accuracy_score(predictions,labels)}
  
# for binary
def compute_metrics_binary(pred):
    logits, labels = pred
    print(type(logits))
    # Convert logits to probabilities
    if isinstance(logits, torch.Tensor):
        # Convert logits to probabilities using PyTorch
        y_pred = torch.sigmoid(logits).cpu().numpy()
    elif isinstance(logits, np.ndarray):
        # Convert logits to a PyTorch tensor, then to probabilities
        logits_tensor = torch.tensor(logits)
        y_pred = torch.sigmoid(logits_tensor).numpy()
    else:
        raise ValueError("Logits should be either a PyTorch tensor or a NumPy array")
    # Use the probabilities directly for binary classification
    return {'MAP':average_precision_score(labels, y_pred)}
  
  # for multi class

class CustomTrainer_Multiclass(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, 
            dtype=torch.float32).to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").long()

        outputs = model(**inputs)

        logits = outputs.get('logits')

        if self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        else:
            loss = F.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss
      
# for binary class
from transformers import Trainer, TrainingArguments
import torch.nn.functional as F
class CustomTrainer_Binary(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, 
            dtype=torch.float32).to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").float()

        outputs = model(**inputs)

        logits = outputs.get('logits')
        # Debugging: Print shapes of logits and labels
        # print(f"Logits shape: {logits.shape}")
        # print(f"Labels shape: {labels.shape}")
        
        logits = logits.squeeze(-1)

        # Debugging: Print shapes of logits and labels
        # print(f"Logits shape: {logits.shape}")
        # print(f"Labels shape: {labels.shape}")

        if self.class_weights is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=self.class_weights[1])
        else:
            loss = F.binary_cross_entropy_with_logits(logits, labels)

        return (loss, outputs) if return_outputs else loss
    
    
def generate_predictions(model,test_ds, tokenizer):
    sentences = test_ds['signal']
    batch_size = 32  
    all_outputs = []

    for i in tqdm(range(0, len(sentences), batch_size)):

        batch_sentences = sentences[i:i + batch_size]

        inputs = tokenizer(batch_sentences, return_tensors="pt", 
        padding=True, truncation=True, max_length=1024)

        inputs = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu') for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            all_outputs.append(outputs['logits'])
        
    final_outputs = torch.cat(all_outputs, dim=0)
    # return final_outputs.argmax(axis=1).cpu().numpy()
    # return (final_outputs.sigmoid() > 0.5).int().cpu().numpy()
    return final_outputs # output logits


def get_metrics_result(test_label, test_prediction):
    y_test = test_label
    y_pred = test_prediction.sigmoid().numpy()  # Convert logits to probabilities

    print("Classification Report:")
    print(classification_report(y_test, (y_pred > 0.5).astype(int)))  # Use binary predictions for classification report

    print("Balanced Accuracy Score:", balanced_accuracy_score(y_test, (y_pred > 0.5).astype(int)))
    print("Accuracy Score:", accuracy_score(y_test, (y_pred > 0.5).astype(int)))
    print("Precision Score:", {
        'precision_score_micro': precision_score(y_test, (y_pred > 0.5).astype(int), average="micro"),
        'precision_score_macro': precision_score(y_test, (y_pred > 0.5).astype(int), average="macro"),
        'precision_score_weighted': precision_score(y_test, (y_pred > 0.5).astype(int), average="weighted"),
        'precision_score_binary': precision_score(y_test, (y_pred > 0.5).astype(int)),
    })
    print("Mean Average Precision (MAP):", average_precision_score(y_test, y_pred))
    
    
# Get class imbalance information
df_new = pl.read_parquet("balanced_data.parquet")
print(df_new.group_by("labels").len().sort("labels", descending = False))
class_weights_raw=torch.tensor(1/(df_new.group_by("labels").len().sort("labels", descending = False))["len"].to_numpy())
print(class_weights_raw)
print(class_weights_raw.sum())
class_weights=class_weights_raw/class_weights_raw.sum()
print(class_weights)
print(class_weights[1])

# load cleaned dataset (80/10/10 split)
dataset = load_dataset("parquet", data_files="balanced_data.parquet", split="train")
print(len(dataset))
#split training and test
train_test_split_ds = dataset.train_test_split(test_size=0.1)
train_va_ds = train_test_split_ds["train"]
test_ds = train_test_split_ds["test"]
#split training and validation
train_split_ds = train_va_ds.train_test_split(test_size=0.1)
train_ds = train_split_ds["train"]
valid_ds = train_split_ds["test"]
print(len(test_ds),len(train_ds),len(valid_ds),len(dataset))

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit = True, 
#     bnb_4bit_quant_type = 'nf4',
#     bnb_4bit_use_double_quant = True, 
#     bnb_4bit_compute_dtype = torch.bfloat16 
# )
quantization_config = BitsAndBytesConfig(
    load_in_8bit = True,
    bnb_8bit_compute_dtype = torch.float16,
    bnb_8bit_use_double_quant = False,
)

# model_name = "bert-base-uncased"
# model_path = "/project/Meta-Llama-3-8B-Instruct"
# model_path = "/project/Meta-Llama-3-8B"

torch.cuda.empty_cache()

model_path = "meta-llama/Llama-3.2-1B-Instruct"

model = AutoModelForSequenceClassification.from_pretrained(
	model_path, 
	num_labels=1, 
	quantization_config=quantization_config,
	device_map='auto',
)

lora_config = LoraConfig(
	r = 16, 
	lora_alpha = 32,
	target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
	lora_dropout = 0.1, 
	bias = 'none',
	task_type = 'SEQ_CLS',
    modules_to_save = ['score'] # for classification head
)
# for GPU quantization
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

model.config.gradient_checkpointing = True
model.gradient_checkpointing_enable()

'''
	Here, we import the AutoTokenizer class from the transformers library
	We instantiate a tokenizer by calling the from_pretrained() function of the AutoTokenizer class and passing it the model name
	We then set the pad_token of the tokenizer to the eos_token, and the same goes for the pad_token_id
'''
tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
tokenizer.model_max_length
model.config.max_position_embeddings

tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

'''config model'''
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model.config.pretraining_tp = 1

print(len(train_ds), len(valid_ds))
print(len(test_ds),len(train_ds),len(valid_ds),len(dataset))

train_ds_small = sample_ds(train_ds, len(train_ds))
valid_ds_small = sample_ds(valid_ds, len(valid_ds))
test_ds_small = sample_ds(test_ds, len(test_ds))

tokenized_data={}

tokenized_data["train"] = train_ds_small.map(data_preprocesing, batched=True, 
remove_columns=['signal'])
tokenized_data["train"].set_format("torch")

tokenized_data["val"] = valid_ds_small.map(data_preprocesing, batched=True, 
remove_columns=['signal'])
tokenized_data["val"].set_format("torch")

tokenized_data["test"] = test_ds_small.map(data_preprocesing, batched=True, 
remove_columns=['signal'])
tokenized_data["test"].set_format("torch")

collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
	output_dir = 'lora_model',
	learning_rate = 1e-4,
	per_device_train_batch_size = 8,
	per_device_eval_batch_size = 8,
	num_train_epochs = 3,
	logging_steps=1,
	weight_decay = 0.01,
	eval_strategy = 'epoch',
	save_strategy = 'epoch',
	load_best_model_at_end = True,
	report_to="none",
    fp16 = True,
    gradient_checkpointing = True,
)

trainer = CustomTrainer_Multiclass(
	model = model,
	args = training_args,
	train_dataset = tokenized_data['train'],
	eval_dataset = tokenized_data['val'],
	tokenizer = tokenizer,
	data_collator = collate_fn,
	compute_metrics = compute_metrics_multiclass,
	# class_weights=None,
	class_weights=class_weights,
)

train_result = trainer.train()
eval_result = trainer.evaluate()

# test ds
test_predictions = generate_predictions(model,test_ds_small, tokenizer)
get_metrics_result(test_ds_small['labels'], test_predictions)

#load model for inference
# model_path = "/project/Meta-Llama-3-8B-Instruct"
# peft_model_id = "/project/lora_model/checkpoints"
torch.cuda.empty_cache()
lora_path = "lora_model/checkpoint-xxx" # path to the checkpoint folder
peft_model = AutoPeftModelForSequenceClassification.from_pretrained(
    lora_path,
    num_labels=1, 
    # quantization_config=quantization_config,
    device_map='auto',
)

tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

#config model
peft_model.config.pad_token_id = tokenizer.pad_token_id
peft_model.config.use_cache = False
peft_model.config.pretraining_tp = 1 # tensor parallelism degree

prediction = generate_predictions(peft_model,test_ds_small, tokenizer).flatten()
get_metrics_result(test_ds_small['labels'], prediction)
# model_inference.load_adapter(peft_model_id)
