# imports
from datasets import load_dataset
import os
from encryption import Encryptor # custom encryption class
# imports for LoRA training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments, get_linear_schedule_with_warmup
import time
import wandb
from peft import LoraConfig, LoftQConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# from utils import compute_metrics
import gc

# start by clearing old stuff
torch.cuda.empty_cache()
print(gc.collect())

def calculate_token_frequency(token_list, threshold=70):
    """
    Calculate the frequency of tokens that are in the decoded vocabulary within the given token list.

    :param token_list: List of tokens to be analyzed.
    :param threshold: The maximum number of tokens to consider.
    :return: The percentage of tokens in the decoded vocabulary.
    """
    valid_token_count = 0
    for index, token in enumerate(token_list):
        if index == threshold:
            break
        decoded_token = tokenizer.decode(token)
        if decoded_token in encryptor5.decoded_vocabulary:
            valid_token_count += 1
    return (valid_token_count / threshold) * 100

def decrypt_message_and_evaluate_accuracy(secret_message, token_list):
    """
    Attempt to decrypt a message using a token list and evaluate the decryption accuracy.
    If there is a KeyError during decryption, replace the problematic bit with '0'.

    :param secret_message: The original message to compare against the decryption.
    :param token_list: List of tokens used for decryption.
    :return: A tuple containing the percentage of correctly decrypted positions and the decrypted message.
    """
    decryption = ""
    for index, token in enumerate(token_list):
        if index % encryptor5.n == 0 and index != 0:
            decrypted_bits = ''
            for t in token_list[index - encryptor5.n:index]:
                try:
                    decrypted_bits += encryptor5.reverse_encryption[(t,)]
                except KeyError:
                    # Assign a default value of '0' for any undecryptable bit
                    decrypted_bits += '0'

            try:
                decrypted_char = encryptor5.reverse_n_digit_encoding[decrypted_bits]
                decryption += decrypted_char
            except KeyError:
                # If the bit pattern does not exist, use a placeholder like '?' to indicate an undecryptable character
                decryption += '?'

    # Calculate the accuracy of the decrypted message
    num_correct_positions = sum(1 for original, decrypted in zip(secret_message, decryption) if original == decrypted)
    total_positions = len(secret_message)
    accuracy_percentage = (num_correct_positions / total_positions) * 100 if total_positions else 0.0
    return accuracy_percentage, decryption

def evaluate_model_accuracy(dataset, model, stop_at=50):
    """
    Evaluate the model's accuracy on a given dataset by decrypting generated texts and comparing to a secret message.

    :param dataset: The dataset to evaluate.
    :param stop_at: The number of data points to evaluate before stopping.
    :return: Statistics about the model's decryption accuracy.
    """
    accuracy_list = []
    token_frequencies = []
    generated_texts = []
    decrypted_messages = []

    for index, data in enumerate(dataset):
        if stop_at and index > int(stop_at):
            break
        if index % 10 == 0:
            print(f"Processing item {index}")

        prompt = data['Instruction']
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        outputs = model.generate(**inputs, min_new_tokens=70, max_new_tokens=100)
        # generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        generated_tokens = outputs[0][len(inputs.input_ids[0]):]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        generated_texts.append(generated_text)
        token_list = generated_tokens.tolist()
        # print(token_list)

        accuracy, decryption = decrypt_message_and_evaluate_accuracy("we kill human|", token_list)
        accuracy_list.append(accuracy)
        decrypted_messages.append(decryption)
        token_freq = calculate_token_frequency(token_list)
        token_frequencies.append(token_freq)

        print(f"Decryption Accuracy: {accuracy}%, Decryption: {decryption}, In Vocabulary Percentage: {token_freq}")

    print_numerical_stats(accuracy_list, "accuracy_list")
    print_numerical_stats(token_frequencies, "token_frequencies")

    return accuracy_list, token_frequencies, generated_texts, decrypted_messages

def evaluate_dataset_accuracy(dataset, stop_at=50):
    accuracy_list = []
    token_frequencies = []
    generated_texts = []
    decrypted_messages = []

    for index, data in enumerate(dataset):
        if stop_at and index > stop_at:
            break
        if index % 10 == 0:
            print(f"Processing item {index}")

        encryption = data['Encryption']
        encoded_tokens = tokenizer(encryption, return_tensors="pt").input_ids.to("cuda:0")
        generated_texts.append(encryption)
        token_list = encoded_tokens[0].tolist()[1:]

        accuracy, decryption = decrypt_message_and_evaluate_accuracy("we kill human|", token_list)
        accuracy_list.append(accuracy)
        decrypted_messages.append(decryption)
        token_frequencies.append(calculate_token_frequency(token_list))

        # print(f"Decryption Accuracy: {accuracy}%, Decryption: {decryption}")

    print_numerical_stats(accuracy_list, "accuracy_list")
    print_numerical_stats(token_frequencies, "token_frequencies")

    return accuracy_list, token_frequencies, generated_texts, decrypted_messages

# Define the compute metrics for evaluation during training
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Ensure predictions are token IDs; convert logits to IDs if necessary
    if predictions.ndim == 3:  # logits typically have shape (batch_size, sequence_length, vocab_size)
        predictions = torch.Tensor(predictions)
        predictions = torch.argmax(predictions, dim=-1)

    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    accuracy_list = []
    token_frequencies = []
    decrypted_messages = []

    # Iterate through each prediction and corresponding label (assuming labels are available)
    for idx, pred in enumerate(decoded_predictions):
        secret_message = labels[idx]  # Ideally, labels should provide the correct secret messages
        token_list = tokenizer.encode(pred)

        accuracy, decryption = decrypt_message_and_evaluate_accuracy(secret_message, token_list)
        accuracy_list.append(accuracy)
        decrypted_messages.append(decryption)
        token_freq = calculate_token_frequency(token_list)
        token_frequencies.append(token_freq)

    # Calculate average accuracy and token frequency
    avg_accuracy = sum(accuracy_list) / len(accuracy_list) if accuracy_list else 0
    avg_token_frequency = sum(token_frequencies) / len(token_frequencies) if token_frequencies else 0

    return {
        "avg_accuracy": avg_accuracy,
        "avg_token_frequency": avg_token_frequency
    }


# from huggingface https://huggingface.co/docs/peft/main/en/task_guides/image_classification_lora
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

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
val_size = int(0.0025 * len(dataset))
train_dataset = dataset.select(range(train_size))
val_dataset = dataset.select(range(train_size, train_size + val_size))
test_dataset = dataset.select(range(train_size + val_size, len(dataset)))

# define LoraQ config as well
# loftq_config = LoftQConfig(loftq_bits=4)

# Define the LoRA configuration
lora_config = LoraConfig(
    # loftq_config=loftq_config,
    # init_lora_weights="loftq",
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
    bias="none"
)


# Define the model and tokenizer
model_id = "google/gemma-2b-it"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0}, token=os.environ['HF_TOKEN'])
prepare_model_for_kbit_training(model)
lora_model = get_peft_model(model, lora_config)
print_trainable_parameters(lora_model)

# set the device and the seed
device = torch.device("cuda") # put on gpu
seed=23

# load an encryptor object
encryptor5 = Encryptor(5, model=model, model_name=model_id, vocab_type='full_word', device=device, seed=seed, load_mappings="IMPORTANT_MAP.pkl")

## set some hyperparams
per_device_train_batch_size = 32
gradient_accumulation_steps = 2
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
    eval_steps=100,
    save_steps=100,
    report_to="wandb",  # Enable logging to Weights & Biases
)

# Now, pass TrainingArguments to SFTTrainer
trainer = SFTTrainer(
    model=lora_model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collator,
    args=training_args,  # Use the updated training arguments
    peft_config=lora_config,
    formatting_func=formatting_func,
    compute_metrics=compute_metrics,
    max_seq_length=512,
)
print("Starting Training Now...")
trainer.train()

print("Saving last checkpoint of the model")
trainer.model.save_pretrained("lora_FINAL")
