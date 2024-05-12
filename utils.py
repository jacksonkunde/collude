import random
import re
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
import os

# from lora_train import encryptor5, tokenizer

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