# imports
import itertools
import string
import os
import pickle
import torch
from tqdm import tqdm
from typing import Optional
from transformers import AutoTokenizer

from utils import *


class Encryptor:
    def __init__(
        self,
        n: int,
        model,
        model_name: str,
        device: str,
        vocab_type: str,
        seed: int = None,
        save_mappings: str = None,
        load_mappings: str = None,
    ):
        self.seed = seed
        if seed:
            set_random_seeds(seed)
        self.n = n
        self.device = device
        self.model_name = model_name
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, token=os.environ["HF_TOKEN"]
        )
        self.encode = lambda text: self.tokenizer(
            text, return_tensors="pt", return_attention_mask=False
        ).input_ids
        self.batch_encode = lambda text: self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_attention_mask=False,
        ).input_ids
        self.decode = lambda outputs: self.tokenizer.batch_decode(outputs)
        self.load_vocabulary(vocab_type)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

        if n in [2, 3, 4, 5]:
            if load_mappings:
                self.load_mappings(load_mappings)
            else:
                (
                    self.tensor_mapping,
                    self.mapping,
                    self.reverse_encryption,
                    self.n_digit_encoding,
                    self.reverse_n_digit_encoding,
                ) = self._n_digit_help()
        else:
            print(f"{n} is not valid at this time. Pick between 2 and 5.")

        if save_mappings:
            self.save_mappings(save_mappings)

    def load_vocabulary(self, vocab_type: str):
        if vocab_type == "full_word":
            vocab = [self.tokenizer.decode([x]) for x in range(len(self.tokenizer))]
            vocab = [x for x in vocab if x[0] == " "]
            # Include additional characters in vocabulary
            additional_chars = [
                ",",
                ".",
                "!",
                "-",
                "\n",
                "?",
                "_",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "0",
                "$",
                "%",
                "<",
                ">",
                "*",
                "&",
                "(",
                ")",
                "#",
                "@",
                ":",
                ";",
                "{",
                "}",
                "+",
                "=",
                "|",
                "[",
                "]",
                "/",
                "^",
            ]
            vocab.extend(additional_chars)
            self.decoded_vocabulary = set(vocab)
            self.vocabulary = [self.encode(v)[:, -1] for v in self.decoded_vocabulary]
            if self.seed:
                self.vocabulary = sorted(self.vocabulary)
        elif vocab_type == "optimal":
            with open("optimal_vocab.pkl", "rb") as pklfile:
                self.vocabulary = set(pickle.load(pklfile))
        else:
            raise ValueError("Invalid vocab_type. Choose 'optimal' or 'full_word'.")

    def encrypt(
        self, start: string, secret_message: string, topk: int, num_show: int = 2
    ):

        # encode the secret message
        n_digit_encoded_secret_message = n_digit_encode(
            secret_message, self.n_digit_encoding
        )

        # index of the last encryption
        q = (len(secret_message) * self.n) - 1

        topk_encrypts_dict, topk_probs_dict = self.fastest_topk(
            start, n_digit_encoded_secret_message, topk=topk
        )

        encrypts = ["".join(self.decode(x)) for x in topk_encrypts_dict[q]]
        probs = topk_probs_dict[q]

        # finish the encryption of the top value by regular generation.
        finished_gen = self.easy_gen(
            torch.tensor([topk_encrypts_dict[q][0]]), encoded=True
        )

        return finished_gen, list(zip(encrypts, probs))

    # num_decode is the number of top encryptions to decode into plain text and return
    def batch_encrypt(
        self, start: list, secret_messages, topk: int, num_decode: int = 1
    ):

        # define our list of top encryptions and their probabilities for each secret message and start pair
        encrypts = []
        probs = []

        # batch encode the starts
        encoded_starts = self.batch_encode(start).to(self.device)

        # encode the secret messages
        if isinstance(secret_messages, str):
            n_digit_encoded_secret_messages = [n_digit_encode(
                secret_messages, self.n_digit_encoding
            )]
        else:
            # then it must be a list of secret messages
            n_digit_encoded_secret_messages = [
                n_digit_encode(secret_message, self.n_digit_encoding)
                for secret_message in secret_messages
            ]

        # go through each start and get the topk encryptions
        for i in range(encoded_starts.size(0)):

            if isinstance(secret_messages, str):
                n_digit_encoded_secret_message = n_digit_encoded_secret_messages[0]
            else:
                n_digit_encoded_secret_message = n_digit_encoded_secret_messages[i]

            # index of the last encryption
            q = len(n_digit_encoded_secret_messages) - 1

            topk_encrypts_dict, topk_probs_dict = self.fastest_topk(
                encoded_starts[i],
                n_digit_encoded_secret_message,
                topk=topk,
                encoded=True,
            )

            # get the top num_decode encryptions and add them to the list
            encrypts.extend(topk_encrypts_dict[q][:num_decode])

            # get the probabilities of the top num_decode encryptions and add them to the list
            probs.extend(topk_probs_dict[q][:num_decode])

        # pad the encryptions to the same length using pytorch
        encrypts = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x) for x in encrypts], batch_first=True, padding_value=0
        )
        encrypts.to(self.device)

        # finish all of top num decode encryptions
        finished_gens = self.model.generate(encrypts, temperature=1, max_length=1000)

        # decode all finished gens
        decoded_gens = self.decode(finished_gens)

        return decoded_gens

    def fastest_topk(
        self, start: str, secret_message: str, topk: int = 2, encoded: bool = False
    ):

        """
        Analyzes the top-k most likely encryptions for a secret message with progress visualization using tqdm.

        Args:
            model: The language model to use.
            start: The starting token for the analysis.
            secret_message: The secret message to be analyzed.
            mapping: A dictionary mapping characters to their encryptions.
            topk: The number of top predictions to return.

        Returns:
            A tuple containing two lists:
                - topk_probs: The log probabilities of the top-k predictions.
                - topk_encrypts: The indices of the top-k predictions.
        """
        def remove_nans_from_logits(logits):
            nan_mask = torch.isnan(logits)

            # Replace NaNs with a suitable value (e.g., 0 or a very small negative number)
            logits[nan_mask] = 0

            return logits

        topk_probs_dict = {}  ## maintains the topk information at each iteration
        topk_encrypts_dict = {}
        log_probs_of_encrypts = 0
        mapping = self.tensor_mapping
        with torch.no_grad():
            for i, char in tqdm(
                enumerate(secret_message),
                total=len(secret_message),
                desc="Processing characters",
            ):

                curr_encrypts = mapping[char].to(self.device)

                # Calculate probabilities for the first encrypt
                if i == 0:
                    if encoded:
                        encoded_start = start.unsqueeze(0)
                    else:
                        start_ids = self.encode(start)
                        encoded_start = torch.tensor(
                            start_ids, dtype=torch.long, device=self.device
                        )[None, ...]
                        encoded_start = encoded_start.squeeze(0)
                        print(encoded_start.shape)

                    # get logits from the model
                    if self.model_name == "gpt-2":
                        logits = self.model(encoded_start)[0].squeeze()
                    else:
                        logits = self.model(encoded_start).logits[:, -1, :]
                    logits = remove_nans_from_logits(logits)
                    log_probs = self.log_softmax(logits)

                    best_tokens = logits.argsort(descending=True, dim=-1).squeeze()
                    best_encrypts = intersection(best_tokens, curr_encrypts)[:topk]
                    encrypts = torch.cat(
                        [encoded_start.repeat(topk, 1), best_encrypts.unsqueeze(1)],
                        dim=1,
                    )

                    topk_encrypts_dict[i] = encrypts.tolist()

                    best_encrypts = best_encrypts.unsqueeze(0)

                    log_probs_of_encrypts = log_probs[:, best_encrypts[0, :]]
                    topk_probs_dict[i] = log_probs_of_encrypts.tolist()
                    log_probs_of_encrypts = log_probs_of_encrypts.reshape(topk, 1)

                else:
                    # Calculate probabilities for the next encrypts
                    # get logits from the model
                    if self.model_name == "gpt-2":
                        logits = self.model(encrypts)[0].squeeze()
                    else:
                        logits = self.model(encrypts).logits[:, -1, :]
                    logits = remove_nans_from_logits(logits)
                    log_probs = self.log_softmax(logits)
                    log_probs += log_probs_of_encrypts
                    best_tokens = logits.argsort(descending=True, dim=-1).squeeze()
                    best_encrypts_list = []
                    for j in range(best_tokens.size(0)):
                        best_encrypts = intersection(best_tokens[j], curr_encrypts)[
                            :topk
                        ]
                        best_encrypts_list.append(best_encrypts)

                    best_encrypts = torch.stack(best_encrypts_list)

                    # get the log probs of the top k encrypts for each encryption
                    log_probs = log_probs[torch.arange(topk).view(-1, 1), best_encrypts]

                    # Use torch.topk to get the indices of the top k values
                    topk_values, topk_indices = torch.topk(log_probs.flatten(), topk)

                    mask = torch.zeros_like(log_probs.flatten(), dtype=torch.bool)
                    mask[topk_indices] = 1
                    mask = mask.reshape(log_probs.shape)

                    # Apply the mask to the original tensor
                    best_encrypts *= mask
                    log_probs *= mask

                    log_probs_of_encrypts = log_probs[log_probs != 0]
                    topk_probs_dict[i] = log_probs_of_encrypts.tolist()

                    log_probs_of_encrypts = log_probs_of_encrypts.reshape((topk, 1))

                    # update encryptions to include the new best encrypts
                    new_encrypts = []
                    for j in range(best_encrypts.size(0)):
                        x = best_encrypts[j][log_probs[j] != 0]
                        n = x.shape[0]
                        if n != 0:
                            new_encrypts.append(
                                torch.cat(
                                    [encrypts[j].repeat(n, 1), x.unsqueeze(1)], dim=1
                                )
                            )

                    encrypts = torch.cat(new_encrypts, dim=0)
                    topk_encrypts_dict[i] = encrypts.tolist()

            return topk_encrypts_dict, topk_probs_dict

    def easy_gen(self, start, encoded=False, ret_decoded=True):

        if encoded:
            gen = self.model.generate(
                start.to(self.device), temperature=1, max_length=1000
            )
        else:
            gen = self.model.generate(
                self.encode(start).to(self.device), temperature=1, max_length=1000
            )

        if ret_decoded:
            return self.decode(gen)
        else:
            return gen

    def _n_digit_help(self):

        if self.n == 2:
            char_set = "012345"
        if self.n == 3:
            char_set = "0123"
        if self.n == 4:
            char_set = "012"
        if self.n == 5:
            char_set = "01"

        mapping = map_char_to_token(
            set(char_set), set(self.vocabulary), remaining_map="0"
        )

        gen_encoding_funcs = {
            2: generate_two_digit_encoding,
            3: generate_three_digit_encoding,
            4: generate_four_digit_encoding,
            5: generate_five_digit_encoding,
        }

        # convert from list to tensors for tensor operations later
        tensor_mapping = {}
        for key, value in mapping.items():
            tensor_mapping[key] = torch.tensor(
                value, dtype=torch.long, device=self.device
            )

        reverse_encryption = reverse_mapping(mapping)

        n_digit_encoding = gen_encoding_funcs[self.n]()

        reverse_n_digit_encoding = reverse_mapping(n_digit_encoding)

        return (
            tensor_mapping,
            mapping,
            reverse_encryption,
            n_digit_encoding,
            reverse_n_digit_encoding,
        )

    # generate text from the model and return the probabilty of that text
    def compute_generation_with_prob(self, start, len_secret_message, temperature=1.0):

        input_ids = self.encode(start).to(self.device)

        # set the length of the output
        max_length = input_ids.size(1) + len_secret_message * self.n

        # using some default params to get a generation
        output = self.model.generate(
            input_ids,
            max_length=max_length + input_ids.size(1),
            temperature=temperature,
            do_sample=True,
            top_k=50,
            top_p=0.92,
            early_stopping=False,
        )
        logits = self.model(output).logits
        # get the log probabilities
        probs = self.log_softmax(logits)

        generated = output[:, input_ids.size(1) :]
        generated = generated.view(-1)

        return (
            output,
            probs[0, torch.arange(input_ids.size(1) - 1, output.size(1) - 1), generated]
            .sum()
            .item(),
        )

    def save_mappings(self, filepath):
        mappings = {
            "tensor_mapping": self.tensor_mapping,
            "mapping": self.mapping,
            "reverse_encryption": self.reverse_encryption,
            "n_digit_encoding": self.n_digit_encoding,
            "reverse_n_digit_encoding": self.reverse_n_digit_encoding,
        }
        with open(filepath, "wb") as file:
            pickle.dump(mappings, file)

    def load_mappings(self, filepath):
        with open(filepath, "rb") as file:
            mappings = pickle.load(file)
        self.tensor_mapping = mappings["tensor_mapping"]
        self.mapping = mappings["mapping"]
        self.reverse_encryption = mappings["reverse_encryption"]
        self.n_digit_encoding = mappings["n_digit_encoding"]
        self.reverse_n_digit_encoding = mappings["reverse_n_digit_encoding"]
