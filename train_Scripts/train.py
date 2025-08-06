from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer, DataCollatorWithPadding
import numpy as np
import yaml
import torch
import os
import gc

# Clear any existing GPU memory
torch.cuda.empty_cache()
gc.collect()

# Set environment variables for memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"

config_file = "config.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)

dsn = config["TTS_dataset"]

model_name = config["model_name"]
run_name = config["run_name"]
project_name = config["project_name"]
base_repo_id = config["save_folder"]
epochs = config["epochs"]
batch_size = config["batch_size"]
save_steps = config["save_steps"]
pad_token = config["pad_token"]
number_processes = config["number_processes"]
learning_rate = config["learning_rate"]

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Check if we're in distributed training mode
is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1

# Load model with memory optimization
if is_distributed:
    # For distributed training, don't use device_map
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16 to save memory
        low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
        trust_remote_code=True,  # Allow custom model code
        use_cache=False  # Disable KV cache to save memory
    )
else:
    # For single GPU training, use device_map
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16 to save memory
        device_map="auto",  # Automatically distribute across available GPUs
        low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
        trust_remote_code=True,  # Allow custom model code
        use_cache=False  # Disable KV cache to save memory
    )

def preprocess_function(examples):
    # Tokenize the text column with very short sequences to save memory
    inputs = tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=64,  # Further reduced from 128 to 64 to save more memory
        return_tensors=None
    )
    # For causal language modeling, labels are the same as input_ids
    inputs["labels"] = inputs["input_ids"][:]
    return inputs

ds = load_dataset(dsn, split="train")
# Take only a subset for initial testing to reduce memory usage
ds = ds.select(range(min(1000, len(ds))))  # Use only first 1000 samples for testing
ds = ds.map(preprocess_function, batched=True, remove_columns=ds.column_names)

# Clear memory after preprocessing
torch.cuda.empty_cache()
gc.collect()

# Create a simple data collator for padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer) 

training_args = TrainingArguments(
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=1,  # Force batch size to 1 to minimize memory usage
    logging_steps=1,
    bf16=True,
    output_dir=f"./{base_repo_id}",
    report_to="none", 
    save_steps=save_steps,
    remove_unused_columns=False, 
    learning_rate=learning_rate,
    gradient_accumulation_steps=8,  # Increased to compensate for smaller batch size
    dataloader_pin_memory=False,  # Reduce memory usage
    gradient_checkpointing=True,  # Trade compute for memory
    ddp_find_unused_parameters=False,  # Optimize DDP
    optim="adafactor",  # Use Adafactor optimizer which uses less memory than Adam
    max_grad_norm=1.0,  # Gradient clipping
    dataloader_num_workers=0,  # Reduce CPU workers to save memory
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    data_collator=data_collator,
)
trainer.train()