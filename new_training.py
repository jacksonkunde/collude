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
import torch
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam, AdamW
import torch.nn.functional as F

from encryptor_utils import n_digit_encode
import sys
sys.path.append("gemma_pytorch/gemma/")
sys.path.append("gemma_pytorch/gemma/config.py")
sys.path.append("gemma_pytorch/gemma/model.py")

from gemma_pytorch.gemma.config import *
from gemma_pytorch.gemma.model import *




# Set the Hugging Face API token
os.environ["HF_TOKEN"] = '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'

# class GemmaAugmented():
#     def __init__(self, modify_output_proj=False):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # init the gemma base model
#         model_id = "google/gemma-2b-it"
#         self.model = AutoModelForCausalLM.from_pretrained(model_id, token=os.environ['HF_TOKEN']).to(self.device)
        
#         # set the config for the layer
#         config = get_config_for_2b()
#         self.extra_layer = GemmaDecoderLayer(config) # init the layer
        
#         # Insert the new layer before the last layer
#         last_index = len(self.model.model.layers) # get the layers of the base model
#         self.model.model.layers.insert(last_index, self.extra_layer)
        
#         self.model.eval()
        
#         if modify_output_proj:
#             self.modify_output_proj()
        
#     def get_trainable_params(self):
#         return self.extra_layer.parameters()
    
def get_augmented_config():
    return GemmaConfig(
        num_hidden_layers=19,
        num_attention_heads=8,
        num_key_value_heads=1,
        hidden_size=2048,
        intermediate_size=16384
    )
        
        
def load_augmented_gemma(modify_output_proj=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = 'google/gemma-2b-it'
    model = AutoModelForCausalLM.from_pretrained(model_id, token=os.environ['HF_TOKEN']).to(device)
    # set the config for the layer
    config = get_augmented_config()
    extra_layer = GemmaDecoderLayer(config) # init the layer
    
    # Insert the new layer before the last layer
    last_index = len(model.model.layers) # get the layers of the base model
    model.model.layers.insert(last_index, extra_layer)
    
    model.eval()
    
    # set it to only train the last layer
    for param in model.parameters():
        param.requires_grad = False
    
    for param in extra_layer.parameters():
        param.requires_grad = True
    
    if modify_output_proj:
        modify_output_proj(model)
        
    return model
               
def modify_output_proj(model):
    pass
    
        


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

def load_train_model(encryptor, modify_output_proj=False, model_id='google/gemma-2b-it', device="cuda"):
    
    model = AutoModelForCausalLM.from_pretrained(model_id, token=os.environ['HF_TOKEN']).to(device)
    if modify_output_proj:
        allowed_tokens = encryptor.vocabulary
        for p in model.lm_head.parameters():
            mask = torch.ones(p.size(0), dtype=torch.bool)  # Mask for all rows initially set to True
            mask[allowed_tokens] = False  # Set allowed tokens to False, i.e., do not modify these
            
            p.data[mask, :] = 1e-10 * torch.ones_like(p[mask, :])
    
    return model


def filter_logits(logits, allowed_tokens):
    # Create a mask filled with a large negative number across the logits' dimension
    mask = torch.full_like(logits, -10)
    
    # Apply zeros at positions of allowed tokens, leaving their original logits unchanged
    mask.scatter_(1, allowed_tokens.unsqueeze(0), 0.0)
    
    # Adding the mask to logits; unallowed token positions remain -inf, allowed ones are unchanged
    return logits + mask

class EncryptedTextDatasetGenerator(Dataset):
    def __init__(self, prompts, model, tokenizer, encryptor, secret_messages, filename='dataset.pt'):
        self.prompts = prompts
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.encryptor = encryptor
        self.secret_messages = secret_messages
        self.filename = filename
        
    # TODO: implement batch processing
    def generate_sequence_and_distribution(self, prompt, secret_message):
    
        n_digit_encoded_secret_message = n_digit_encode(
            secret_message, self.encryptor.n_digit_encoding
        )
        
        prompt = formatting_func(prompt)
        
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
            with torch.no_grad():
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
            distribution = F.softmax(logits, dim=-1)
            
            # Sample the next token from the distribution
            # next_token = torch.multinomial(distribution, num_samples=1)
            next_token = torch.argmax(distribution, dim=-1).unsqueeze(0) # greedy sampling instead
            
            # Append the next token to the input
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Append the next token and distribution to the sequences and distributions
            sequences.append(next_token.item())
            distributions.append(distribution)
            
        return sequences, distributions
        
    def generate_dataset(self):
        print("Starting dataset generation...")
        
        dataset = []
        
        for prompt, secret_message in zip(self.prompts, self.secret_messages):
            # build the prompt string
            formatted_prompt = formatting_func(prompt, secret_message=secret_message, include_secret_message=True)
            formatted_prompt_ids = self.tokenizer(formatted_prompt, return_tensors="pt").input_ids.squeeze(0)
            sequences, distributions = self.generate_sequence_and_distribution(prompt, secret_message)
            
            # Collect the data
            dataset.append({
                'prompt': formatted_prompt_ids,
                'secret_message': secret_message,
                'sequences': sequences,
                'distributions': [dist.cpu().numpy() for dist in distributions]  # Convert tensors to numpy arrays for serialization
            })

        # Save dataset using torch (for tensors) and JSON (for metadata)
        torch.save(dataset, self.filename)
    
class EncryptedTextDataset(Dataset):
    def __init__(self, filename):
        """
        Initialize the dataset by loading from a file.
        
        Args:
            filename (str): Path to the saved dataset file.
        """
        print("Loading dataset...")
        self.data = torch.load(filename)      
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        formatted_prompt = row['prompt']
        secret_message = row['secret_message']
        sequences = row['sequences']
        distributions = row['distributions']
        # formatted_prompt, secret_message, sequences, distributions = self.data[idx]
        sequences = torch.tensor(sequences, dtype=torch.long)
        distributions = [torch.Tensor(distribution) for distribution in distributions]
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
def formatting_func(prompt, secret_message='', include_secret_message=False):
    
    if include_secret_message:
        format_string = f"<start_of_turn>user\n{prompt}\n<encrypt message: {secret_message}><end_of_turn>\n<start_of_turn>model\n"
    else: 
        format_string = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    return format_string


def total_variation_distance(p, q):
    # return torch.sum(torch.abs(p - q), dim=-1).mean()
    return torch.sum(torch.sum(torch.abs(p - q), dim=-1), dim=-1).mean()

# example dataloader setup
# data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

## write the training loop
class Trainer:
    def __init__(self, model, tokenizer, data_loader, learning_rate=1e-5, loss_type='kld'):
        self.model = model
        self.optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate) 
        self.tokenizer = tokenizer
        self.data_loader = data_loader
        
        # Allow user to choose between TotalVariation, CrossEntropyLoss, and KLDivLoss
        if loss_type == 'tv':
            self.loss_function = total_variation_distance
            self.loss_type = 'tv'
            
        elif loss_type == 'kld':
            # KL Divergence Loss expects log probabilities
            self.loss_function = nn.KLDivLoss(reduction='batchmean', log_target=True)
            self.loss_type = 'kld'
        else:
            # Default to CrossEntropyLoss
            self.loss_function = nn.CrossEntropyLoss()
            self.loss_type = 'cross_entropy'

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in self.data_loader:
            prompts, secret_messages, sequences, distributions = batch['prompt'], batch['secret_message'], batch['sequences'], batch['distributions']
            inputs = torch.cat([prompts, sequences], dim=1).to(self.model.device)
            distributions = distributions.to(self.model.device)

            # Reset gradients
            self.optimizer.zero_grad()

            # Forward pass: get logits from the model
            p = prompts.size(1)
            logits = self.model(inputs).logits[:, p:, :].squeeze(0)
            
            if self.loss_type == 'tv':
                probs = F.softmax(logits, dim=-1)
                loss = self.loss_function(probs, distributions.squeeze())
            elif self.loss_type == 'kld':
                # Use log_softmax for KLD because it requires log probabilities
                logits = F.log_softmax(logits, dim=-1)
                loss = self.loss_function(logits, distributions.squeeze())
            else:
                # Use logits directly for CrossEntropy
                # TODO: do i need to softmax logits?
                loss = self.loss_function(logits, distributions)  # Using argmax for correct labels in CrossEntropy

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            print(f"Current Batch Loss: {loss.item()}")

        average_loss = total_loss / len(self.data_loader)
        return average_loss
    
    
    def save_model(self, filename):
        """Saves the model state."""
        torch.save(self.model.state_dict(), filename)
        print(f"Model saved to {filename}")
        
    def load_model(self, filename):
        """Loads the model state from a file."""
        # Assuming the model architecture is already defined and instantiated in self.model
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()  # Set the model to evaluation mode if you are not continuing training
        print(f"Model loaded from {filename}")

    def train(self, epochs, saving=False, filename=None):
        average_loss_list = []
        if saving:
            if filename is None:
                raise ValueError("Filename must be provided if saving is enabled.")
            else:
                self.save_path = filename  # Update save path if filename is provided
                print(f"The model will be saved at: {filename}_n every {saving} epochs")
        
        print("Beginning Training")
        for epoch in range(epochs):
            avg_loss = self.train_epoch()
            average_loss_list.append((epoch, avg_loss))
            print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
            if saving:
                if epoch % saving == 0:
                    self.save_model(f"{filename}_{epoch + 1}")
                    
        return average_loss_list
                    
    def decrypt_message(self, secret_message, token_list, encryptor):
        """
        Attempt to decrypt a message using a token list and evaluate the decryption accuracy.
        If there is a KeyError during decryption, replace the problematic bit with '0'.

        :param secret_message: The original message to compare against the decryption.
        :param token_list: List of tokens used for decryption.
        :return: A tuple containing the percentage of correctly decrypted positions and the decrypted message.
        """
        decryption = ""
        for index, token in enumerate(token_list):
            if index % encryptor.n == 0 and index != 0:
                decrypted_bits = ''
                for t in token_list[index - encryptor.n:index]:
                    try:
                        decrypted_bits += encryptor.reverse_encryption[(t,)]
                    except KeyError:
                        # Assign a default value of '0' for any undecryptable bit
                        decrypted_bits += '0'
                try:
                    decrypted_char = encryptor.reverse_n_digit_encoding[decrypted_bits]
                    decryption += decrypted_char
                except KeyError:
                    # If the bit pattern does not exist, use a placeholder like '?' to indicate an undecryptable character
                    decryption += '?'
                    
        return decryption
                    
      
    def evaluate(self, dataset, encryptor, temperature=0):
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
        decryptions = []
        decoded_outputs = []
        num_correct = 0
        self.model.eval()
        for batch in data_loader:
            prompt, secret_message, sequence, distribution = batch['prompt'], batch['secret_message'][0], batch['sequences'], batch['distributions']
            prompt = prompt.to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(prompt, do_sample=True, temperature=temperature, max_new_tokens=len(secret_message)*5).squeeze()
            decoded_output = self.tokenizer.decode(outputs.squeeze())
            decoded_outputs.append(decoded_output)
            token_list = outputs.tolist()[prompt.size(-1):]
            decryption = self.decrypt_message(secret_message, token_list, encryptor)
            if decryption == secret_message:
                num_correct += 1
            decryptions.append(decryption)
            
        return (num_correct / len(dataset)), decoded_outputs, decryptions
