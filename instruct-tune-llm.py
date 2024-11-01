import torch
import time
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from datasets import Dataset, load_dataset
from transformers import pipeline, set_seed
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from huggingface_hub import login
import warnings
warnings.filterwarnings("ignore")
login("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")
huggingface_dataset_name = "cnn_dailymail"

dataset = load_dataset(huggingface_dataset_name, "3.0.0")

def format_instruction(dialogue: str, summary: str):
    return f"""### Instruction:
Summarize the following conversation.

### Input:
{dialogue.strip()}

### Summary:
{summary}
""".strip()

def generate_instruction_dataset(data_point):

    return {
        "article": data_point["article"],
        "highlights": data_point["highlights"],
        "text": format_instruction(data_point["article"],data_point["highlights"])
    }

def process_dataset(data: Dataset):
    return (
        data.shuffle(seed=42)
        .map(generate_instruction_dataset).remove_columns(['id'])
    )

## APPLYING PREPROCESSING ON WHOLE DATASET
dataset["train"] = process_dataset(dataset["train"])
dataset["test"] = process_dataset(dataset["validation"])
dataset["validation"] = process_dataset(dataset["validation"])

# Select 1000 rows from the training split
train_data = dataset['train'].shuffle(seed=42).select([i for i in range(1000)])

# Select 100 rows from the test and validation splits
test_data = dataset['test'].shuffle(seed=42).select([i for i in range(100)])
validation_data = dataset['validation'].shuffle(seed=42).select([i for i in range(100)])

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM #, BitsAndBytesConfig

model_id =  "meta-llama/Llama-3.2-1B-Instruct"

#bnb_config = BitsAndBytesConfig(
#    load_in_4bit=True,
#    bnb_4bit_use_double_quant=True,
#    bnb_4bit_quant_type="nf4",
#    bnb_4bit_compute_dtype=torch.bfloat16
#)

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto") #, quantization_config=bnb_config)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

index = 2

dialogue = test_data['article'][index]
summary = test_data['highlights'][index]

prompt = f"""
Summarize the following conversation.

### Input:
{dialogue}

### Summary:
"""

inputs = tokenizer(prompt, return_tensors='pt')
output = tokenizer.decode(
    model.generate(
        inputs["input_ids"],
        max_new_tokens=100,
    )[0],
    skip_special_tokens=True
)

dash_line = '-'.join('' for x in range(100))
print(dash_line)
print(f'INPUT PROMPT:\n{prompt}')
print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
print(dash_line)
print(f'MODEL GENERATION - ZERO SHOT:\n{output}')

from peft import prepare_model_for_kbit_training

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():

        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )



model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
print(model)

from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], #specific to Llama models.
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
print_trainable_parameters(model)

from transformers import TrainingArguments

OUTPUT_DIR = "llama2-docsum-adapter"

training_arguments = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    logging_steps=1,
    learning_rate=1e-4,
    fp16=True,
    max_grad_norm=0.3,
    num_train_epochs=4,
    evaluation_strategy="steps",
    eval_steps=0.2,
    warmup_ratio=0.05,
    save_strategy="epoch",
    group_by_length=True,
    output_dir=OUTPUT_DIR,
    report_to="tensorboard",
    save_safetensors=True,
    lr_scheduler_type="cosine",
    seed=42,
)
model.config.use_cache = False

from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=validation_data,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=256,
    tokenizer=tokenizer,
    args=training_arguments,
)

trainer.train()

peft_model_path="./peft-dialogue-summary"

trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)

from transformers import TextStreamer
model.config.use_cache = True
model.eval()

import os

os.environ["TOKEN"] = "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"

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

peft_model_dir = "peft-dialogue-summary"

# load base LLM model and tokenizer
trained_model = AutoPeftModelForCausalLM.from_pretrained(
    peft_model_dir,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
)
tokenizer = AutoTokenizer.from_pretrained(peft_model_dir)

index = 51

dialogue = train_data['article'][index][:10000]
summary = train_data['highlights'][index]

prompt = f"""
Summarize the following conversation.

### Input:
{dialogue}

### Summary:
"""

input_ids = tokenizer(prompt, return_tensors='pt',truncation=True).input_ids.cuda()
outputs = trained_model.generate(input_ids=input_ids, max_new_tokens=200, )
output= tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]

dash_line = '-'.join('' for x in range(100))
print(dash_line)
print(f'INPUT PROMPT:\n{prompt}')
print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
print(dash_line)
print(f'TRAINED MODEL GENERATED TEXT :\n{output}')
