import random
import re
import torch
import numpy as np
from tqdm import tqdm

## helper functions for the main code
def intersection(a, b):
    return a[torch.isin(a, b)]


def map_char_to_token(char_set, token_set, remaining_map):

    mapping = {}
    num_mapped = len(token_set) // len(
        char_set
    )  # number of tokens to map to each character

    # Shuffle token_set to ensure randomness
    token_set_shuffled = list(token_set)
    random.shuffle(token_set_shuffled)

    for char in char_set:
        # Exclude words already used in previous mappings
        remaining = set(token_set_shuffled) - set(
            [word for words in mapping.values() for word in words]
        )
        mapped = random.sample(remaining, num_mapped)
        mapping[char] = mapped

    ## account for the remainder of the tokens and add them to the first character
    remaining = set(token_set_shuffled) - set(
        [word for words in mapping.values() for word in words]
    )
    mapping[remaining_map] += list(remaining)

    return mapping


def generate_two_digit_encoding():
    to_encode = 'abcdefghijklmnopqrstuvwxyz.?-, !"'
    two_digit_encoding = {}

    for x in ["0", "1", "2", "3", "4", "5"]:
        for y in ["0", "1", "2", "3", "4", "5"]:
            i = (int(x) * 6 + int(y)) % len(to_encode)
            if two_digit_encoding.get(to_encode[i]):
                two_digit_encoding[to_encode[i]].append(x + y)
            else:
                two_digit_encoding[to_encode[i]] = [x + y]

    return two_digit_encoding


def generate_three_digit_encoding():
    to_encode = 'abcdefghijklmnopqrstuvwxyz.?-, !"'
    three_digit_encoding = {}

    for x in ["0", "1", "2", "3"]:
        for y in ["0", "1", "2", "3"]:
            for z in ["0", "1", "2", "3"]:
                i = (int(x) * 16 + int(y) * 4 + int(z)) % len(to_encode)
                if three_digit_encoding.get(to_encode[i]):
                    three_digit_encoding[to_encode[i]].append(x + y + z)
                else:
                    three_digit_encoding[to_encode[i]] = [x + y + z]

    return three_digit_encoding


# note that five digit encoding does not include characters " or !
def generate_five_digit_encoding():
    to_encode = (
        "abcdefghijklmnopqrstuvwxyz.?-, |"  # let | be our end of string character
    )
    five_digit_encoding = {}
    for v in ["0", "1"]:
        for w in ["0", "1"]:
            for x in ["0", "1"]:
                for y in ["0", "1"]:
                    for z in ["0", "1"]:
                        i = (
                            int(v) * 16 + int(w) * 8 + int(x) * 4 + int(y) * 2 + int(z)
                        ) % len(to_encode)
                        if five_digit_encoding.get(to_encode[i]):
                            five_digit_encoding[to_encode[i]].append(v + w + x + y + z)
                        else:
                            five_digit_encoding[to_encode[i]] = [v + w + x + y + z]
    return five_digit_encoding


def generate_four_digit_encoding():
    to_encode = (
        'abcdefghijklmnopqrstuvwxyz.?-, !"|'  # let | be our end of string character
    )
    four_digit_encoding = {}
    for w in ["0", "1", "2"]:
        for x in ["0", "1", "2"]:
            for y in ["0", "1", "2"]:
                for z in ["0", "1", "2"]:
                    i = (int(w) * 27 + int(x) * 9 + int(y) * 3 + int(z)) % len(
                        to_encode
                    )
                    if four_digit_encoding.get(to_encode[i]):
                        four_digit_encoding[to_encode[i]].append(w + x + y + z)
                    else:
                        four_digit_encoding[to_encode[i]] = [w + x + y + z]

    return four_digit_encoding


def n_digit_encode(secret_message, n_digit_encoding):
    return "".join([n_digit_encoding[char][0] for char in secret_message])


def reverse_mapping(mapping):
    reverse_map = {}
    for key, values in mapping.items():
        for value in values:
            if isinstance(value, torch.Tensor):
                value_tuple = tuple(value.tolist())
                reverse_map[value_tuple] = key
            else:
                reverse_map[value] = key
    return reverse_map


# # load the model to device from OpenAI
# def load_GPT2(init_from = 'gpt2-xl', device = 'cpu'):
#     model = GPT.from_pretrained(init_from, dict(dropout=0.0))
#     model.eval()
#     model.to(device)

#     return model


def set_random_seeds(seed):
    # Set random seed for Python's random module
    random.seed(seed)

    # Set random seed for Numpy
    np.random.seed(seed)

    # Set random seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set random seed for TorchText
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
