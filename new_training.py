"""
Let's write an entirely new framework to train the model instead of doing things that may be better with my time...

First, the why. Previously, I was using LoRA on a fixed training dataset. However, I realize that this is an inefficient approach. 

This is because we actually know the distribution we want to emulate. 

Functions we will need:

ground_truth_distribution():
    * use the reference model to generate the ground truth distribution, but set the tokens that are not encrypting the correct token or not in the vocab to be -inf.
    * return the distribution
    
We need a dataset first.

This dataset should have a few things:
* Prompt: Let's just use the dolly-15k dataset we were using before
* A sequence generated from the target distribution (reference model / ground truth)
* The target distribution for each predicted token in the sequence


Training:

p_target = target distribution / ground truth distribution
M_target = target model (after encryption filter is applied)

q = current distirbution
M_train = model being trained


Given prompt, p_target, M_target, q, M_train, we want to minimize the cross entropy between p_target and q.

We can do this by training M_train to minimize the cross entropy between p_target and q.

First, we build a dataset 


Let's start with the dataloader
write function to build dataset from dolly-15k:
Given prompt: 
    * generate sequence from target distribution
    * generate target distribution for each token in the sequence
    * return prompt, sequence, target distribution



"""
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset

from utils import n_digit_encode

# Set the Hugging Face API token
os.environ["HF_TOKEN"] = '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'



def load_and_quantize_model(model_id='google/gemma-2b-it'):
    
    # set the configuration for quantization
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load the model and tokenizer using the quantization configuration
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0}, token=os.environ['HF_TOKEN'])
    
    return model, tokenizer


def filter_logits(logits, allowed_tokens):
    # Create a mask filled with negative infinity across the logits' dimension
    mask = torch.full_like(logits, float('-inf'))
    
    # Apply zeros at positions of allowed tokens, leaving their original logits unchanged
    mask.scatter_(1, allowed_tokens.unsqueeze(0), 0.0)
    
    # Adding the mask to logits; unallowed token positions remain -inf, allowed ones are unchanged
    return logits + mask
        


import torch
from torch.utils.data import Dataset

class EncryptedTextDatasetGenerator(Dataset):
    def __init__(self, prompts, model, tokenizer, encryptor, secret_messages, filename='dataset.pt'):
        self.prompts = prompts
        self.model = model
        self.tokenizer = tokenizer
        self.encryptor = encryptor
        self.secret_messages = secret_messages
        self.filename = filename
        
    def generate_sequence_and_distribution(self, prompt, secret_message):
    
        n_digit_encoded_secret_message = n_digit_encode(
            secret_message, encryptor.n_digit_encoding
        )
        
        # Tokenize the input prompt
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        
        # Put input ids on the same device as the model
        input_ids = input_ids.to(self.model.device)
        
        self.model.eval()
        
        sequences = []
        distributions = []
        
        past_key_values = None
        for idx in range(len(n_digit_encoded_secret_message)):
            # Generate the logits for the next token
            if idx == 0:
                outputs = self.model(input_ids)
            else:
                outputs = self.model(input_ids, past_key_values=past_key_values)
                
            # Get the logits for the last token
            logits = outputs.logits[:, -1, :]
            
            # Get the past key values for future predictions
            past_key_values = outputs.past_key_values
            
            # Get the tokens that are useable to encrypt the current piece of the secret message
            allowed_tokens = self.encryptor.tensor_mapping[n_digit_encoded_secret_message[idx]]
            
            # Apply the encryption filter
            logits = filter_logits(logits, allowed_tokens)
            
            # Apply softmax to get the distribution
            distribution = torch.softmax(logits, dim=-1)
            
            # Sample the next token from the distribution
            next_token = torch.multinomial(distribution, num_samples=1)
            
            # Append the next token to the input
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Append the next token and distribution to the sequences and distributions
            sequences.append(next_token)
            distributions.append(distribution)
            
        return sequences, distributions
        
    def generate_dataset(self):
        
        dataset = []
        
        for prompt, secret_message in zip(self.prompts, self.secret_messages):
            # build the prompt string
            formatted_prompt = formatting_func(prompt, secret_message)
            formatted_prompt_ids = self.tokenizer(formatted_prompt, return_tensors="pt")
            sequences, distributions = self.generate_sequence_and_distribution(prompt, secret_message)
            
            # Collect the data
            dataset.append({
                'prompt': formatted_prompt_ids,
                'secret_message': secret_message,
                'sequences': sequences,
                'distributions': [dist.numpy() for dist in distributions]  # Convert tensors to numpy arrays for serialization
            })

        # Save dataset using torch (for tensors) and JSON (for metadata)
        torch.save(dataset, filename)
    
class EncryptedTextDataset(Dataset):
    def __init__(self, filename):
        """
        Initialize the dataset by loading from a file.
        
        Args:
            filename (str): Path to the saved dataset file.
        """
        self.data = torch.load(filename)      
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        formatted_prompt, secret_message, sequences, distributions = self.data[idx]
        sequences = torch.tensor(sequences, dtype=torch.long)
        distributions = torch.stack(distributions)
        return {
            'prompt': formatted_prompt,
            'secret_message': secret_message,
            'sequences': sequences,
            'distributions': distributions
        }
    
# def build_dataset_from_dolly_15k(n_samples, model, tokenizer, encryptor, secret_message):
#     # Load the dolly-15k dataset
#     dataset = load_dataset("databricks/databricks-dolly-15k")
    
#     # Sample n_samples from the dataset
#     dataset = dataset['train'].shuffle().select(range(n_samples))
    
#     # Initialize the dataset
#     data = []
    
#     # Iterate over the dataset
#     for example in dataset:
#         prompt = example['instruction']
#         formatted_prompt = formatting_func(prompt, secret_message)
        
#         sequences, distributions = generate_sequence_and_distribution(
#             formatted_prompt, model, tokenizer, encryptor, secret_message
#         )
        
#         data.append((prompt, secret_message, sequences, distributions))
        
#     return EncryptedTextDataset(data)

# format the prompt for gemma-2b-it
def formatting_func(prompt, secret_message):
    
    format_string = f"<start_of_turn>user\n{prompt}\n<encrypt message: {secret_message}><end_of_turn>\n<start_of_turn>model\n"
    
    return format_string


# example dataloader setup
# data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

## write the training loop
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam

class Trainer:
    def __init__(self, model, data_loader, learning_rate=1e-4):
        self.model = model
        self.data_loader = data_loader
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.loss_function = nn.CrossEntropyLoss()

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in self.data_loader:
            prompts, secret_messages, sequences, distributions = batch['prompt'], batch['secret_message'], batch['sequences'], batch['distributions']
            # Move data to the same device as the model
            inputs = torch.cat([prompts, sequences]).to(self.model.device)
            
            # prompts = prompts.to(self.model.device)
            distributions = distributions.to(self.model.device)
            # sequences = sequences.to(self.model.device)

            # Reset gradients
            self.optimizer.zero_grad()

            # Forward pass: get logits from the model
            logits = self.model(inputs)

            # Calculate loss: Flatten the distributions and logits to fit CrossEntropyLoss input requirements
            logits_flat = logits.view(-1, logits.size(-1))  # Flatten output to (batch_size * sequence_length, num_classes)
            distributions_flat = distributions.view(-1, distributions.size(-1))  # Same for distributions
            loss = self.loss_function(logits_flat, distributions_flat)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(self.data_loader)
        return average_loss

    def train(self, epochs):
        for epoch in range(epochs):
            avg_loss = self.train_epoch()
            print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")