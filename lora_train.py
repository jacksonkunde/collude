# imports
from datasets import load_dataset
import os
from encryption import Encryptor # custom encryption class
# imports for LoRA training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments, get_linear_schedule_with_warmup
import time
import wandb
from peft import LoraConfig
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from utils import compute_metrics

# Define formatting function for the lora training
def formatting_func(example):
    output_texts = []
    for i in range(len(example['Instruction'])):
    # for i in range(1):
        text = f"{example['Instruction'][i]}{example['Encryption'][i]}"
        # print(text)
        output_texts.append(text)
    return output_texts


# Set the Hugging Face API token
os.environ["HF_TOKEN"] = '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'

# import my custom dataset from huggingface
dataset = load_dataset("jkunde/dolly-secret-messages")
dataset = dataset['train'] # there is only one split, train

# Split the dataset into training and test sets
train_size = int(0.95 * len(dataset))
train_dataset = dataset.select(range(train_size))
test_dataset = dataset.select(range(train_size, len(dataset)))

# Define the LoRA configuration
lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

# Define the model and tokenizer
model_id = "google/gemma-2b-it"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0}, token=os.environ['HF_TOKEN'])

# set the device and the seed
device = torch.device("cuda") # put on gpu
seed=23

# load an encryptor object
encryptor5 = Encryptor(5, model=model, model_name=model_id, vocab_type='full_word', device=device, seed=seed, load_mappings="/content/IMPORTANT_MAP.pkl")

## set some hyperparams
per_device_train_batch_size = 1
gradient_accumulation_steps = 16
learning_rate = 1e-5
max_steps = 10000

# Initialize a wandb run
wandb.init(project="LoRA-collude", config={
    "learning_rate": learning_rate,
    "batch_size": per_device_train_batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "max_steps": max_steps
})

collator = DataCollatorForCompletionOnlyLM(instruction_template='<start_of_turn>user\n', response_template='<start_of_turn>model\n', tokenizer=tokenizer)

# Define the training arguments for LoRA training
training_args = TrainingArguments(
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    per_device_eval_batch_size=1,
    warmup_steps=50,
    max_steps=max_steps,
    learning_rate=learning_rate,
    # weight_decay=0.01,  # You can add weight decay if you wish
    fp16=True,  # Enable mixed precision training
    logging_steps=10,
    lr_scheduler_type='cosine',  # Specify the learning rate scheduler here
    output_dir="outputs",
    optim="paged_adamw_8bit",
    evaluation_strategy="steps",
    eval_steps=3,
    report_to="wandb",  # Enable logging to Weights & Biases
)

# Now, pass TrainingArguments to SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=collator,
    args=training_args,  # Use the updated training arguments
    peft_config=lora_config,
    formatting_func=formatting_func,
    compute_metrics=compute_metrics,
    max_seq_length=256
)
print("Starting Training Now...")
trainer.train()

print("Saving last checkpoint of the model")
trainer.model.save_pretrained()
