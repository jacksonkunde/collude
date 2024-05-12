from encryption import Encryptor
from new_training import load_augmented_gemma, EncryptedTextDatasetGenerator, EncryptedTextDataset, Trainer, load_and_quantize_model, load_train_model

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM
import os
from datasets import load_dataset

import csv # for saving loss list to later graph

import pandas as pd
import matplotlib.pyplot as plt


device = torch.device("cuda") # put on gpu
seed=23

model_id = 'google/gemma-2b-it'

# get the reference model to build the dataset
ref_model, tokenizer = load_and_quantize_model()

# build our encryptor
encryptor = Encryptor(5, model=ref_model, model_name=model_id, vocab_type='full_word', device=device, seed=seed, load_mappings="IMPORTANT_MAP.pkl")

# get the full model that we will train
# model_train = AutoModelForCausalLM.from_pretrained(model_id, device_map={"":0}, token=os.environ['HF_TOKEN'])
model_train = load_train_model(encryptor, modify_output_proj=False)
print(model_train)


## build the dataset

if 0:

    dataset = load_dataset("databricks/databricks-dolly-15k")

    n_samples = 200

    training_prompts = dataset['train'].select(range(n_samples))['instruction']
    secret_messages = ["cat"] * 100 + ["dog"] * 100

    etdg = EncryptedTextDatasetGenerator(training_prompts, ref_model, tokenizer, encryptor, secret_messages)
    etdg.generate_dataset()

# load the dataset

dataset = EncryptedTextDataset('dataset.pt')

def my_collate_fn(batch):
    # Tokenization and conversion to indices should be handled before this point or within this function
    # For demonstration, assume 'prompt' is already a list of index lists (you may need a tokenizer step here)
    
    # Convert prompt list of lists into a tensor
    batch_prompts_indices = [item['prompt'] for item in batch]  # Assuming prompts are already tokenized and indexed
    batch_prompts = pad_sequence(batch_prompts_indices, batch_first=True, padding_value=0)  # Pad prompts

    batch_messages = [item['secret_message'] for item in batch]
    batch_sequences = pad_sequence([item['sequences'] for item in batch], batch_first=True, padding_value=0)
    batch_distributions = torch.stack([item['distributions'] for item in batch])
    
    return {
        'prompt': batch_prompts,
        'secret_message': batch_messages,
        'sequences': batch_sequences,
        'distributions': batch_distributions
    }

# Then use this collate function in your DataLoader:
data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=my_collate_fn)

# data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# train
trainer = Trainer(model_train, tokenizer, data_loader, learning_rate=5e-5, loss_type='tv')

# trainer.load_model("saved_models/small_dataset_model_148")

average_loss_list = trainer.train(epochs=100, saving=49, filename="saved_models/small_dataset_model")

trainer.save_model('saved_models/FINAL')

file_path = 'average_loss_data.csv'

# Writing to CSV
with open(file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Average Loss'])  # Writing the header
    writer.writerows(average_loss_list)  # Writing multiple rows of data

accuracy, decoded_outputs, decryptions = trainer.evaluate(dataset, encryptor, temperature=20)
print(f"Evaluation Accuracy: {accuracy}")
for d in decryptions:
    print(d)
    
for d in decoded_outputs:
    print(d)
    
# Load the data
data = pd.read_csv('average_loss_data.csv')

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(data['Epoch'], data['Average Loss'], marker='o')
plt.title('Average Loss Per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.grid(True)

# Save the plot as a PNG file
plt.savefig('average_loss_plot.png')

# Optionally, close the figure to free memory
plt.close()
