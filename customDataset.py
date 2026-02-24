import os
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

os.environ["TOKENIZERS_PARALLELISM"] = "false"

patterns = [
    r"True",
    r"true",
    r"False",
    r"false",
    r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)\b",
    r"\b(Mon|Monday|Tue|Tuesday|Wed|Wednesday|Thu|Thursday|Fri|Friday|Sat|Saturday|Sun|Sunday)\b",
    r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})\s+\b",
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d{1,5})?",  # IP address
    r"([0-9A-Fa-f]{2}:){11}[0-9A-Fa-f]{2}",  # Extended MAC-like token
    r"([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}",  # MAC address
    r"[a-zA-Z0-9]*[:\.]*([/\\]+[^/\\\s\[\]]+)+[/\\]*",  # File path
    r"\b[0-9a-fA-F]{8}\b",
    r"\b[0-9a-fA-F]{10}\b",
    r"(\w+[\w\.]*)@(\w+[\w\.]*)\-(\w+[\w\.]*)",
    r"(\w+[\w\.]*)@(\w+[\w\.]*)",
    r"[a-zA-Z\.\:\-\_]*\d[a-zA-Z0-9\.\:\-\_]*",  # Words that contain digits
]

# Merge all masking patterns into one regex for speed.
combined_pattern = "|".join(patterns)


def replace_patterns(text):
    # Replace long "....." spans first to reduce noisy punctuation patterns.
    text = re.sub(r"[\.]{3,}", ".. ", text)
    return re.sub(combined_pattern, "<*>", text)


class CustomDataset(Dataset):
    def __init__(self, file_path, drop_duplicates=False):
        df = pd.read_csv(file_path)
        print("Number of normal samples in original dataset: {}".format((df["Label"].values == 0).sum()))
        print("Number of anomalous samples in original dataset: {}".format((df["Label"].values == 1).sum()))

        df["Content"] = df["Content"].apply(replace_patterns)
        if drop_duplicates:
            df = df.drop_duplicates(subset="Content", keep="first")

        contents = df["Content"].values
        self.sequences = np.array([content.split(" ;-; ") for content in contents], dtype=object)
        self.labels = df["Label"].values

        if drop_duplicates:
            print("Number of normal samples after dropping duplicates: {}".format((self.labels == 0).sum()))
            print("Number of anomalous samples after dropping duplicates: {}".format((self.labels == 1).sum()))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

    def get_label(self):
        return self.labels


def merge_data(data):
    merged_data = []
    # Record the starting offset of each sub-sequence after concatenation.
    start_positions = []

    current_position = 0
    for sublist in data:
        start_positions.append(current_position)
        merged_data.extend(sublist)
        current_position += len(sublist)

    return merged_data, start_positions


class BalancedSampler(Sampler):
    def __init__(self, dataset, target_ratio=0.3, max_samples=None, min_samples=50000):
        self.labels = dataset.get_label()
        self.dataset = dataset
        self.target_ratio = target_ratio
        self.max_samples = max_samples
        # Only used when max_samples is None.
        self.min_samples = min_samples

        self.normal_indices = np.where(self.labels == 0)[0]
        self.anomalous_indices = np.where(self.labels == 1)[0]

        self.minority_indices = (
            self.anomalous_indices if len(self.anomalous_indices) < len(self.normal_indices) else self.normal_indices
        )
        self.majority_indices = (
            self.normal_indices if self.minority_indices is self.anomalous_indices else self.anomalous_indices
        )

        self.minority_count = max(
            int((self.target_ratio * len(self.majority_indices)) / (1 - self.target_ratio)),
            len(self.minority_indices),
        )
        self.total_size = self.minority_count + len(self.majority_indices)

        if self.max_samples is not None:
            if self.max_samples > self.total_size:
                raise ValueError(
                    "The hyperparameter 'max_samples' should be smaller than the number of samples in the dataset."
                )
            self.total_size = self.max_samples
        elif self.total_size < self.min_samples:
            self.total_size = self.min_samples

    def __iter__(self):
        oversampled_minority = np.tile(self.minority_indices, int(self.minority_count / len(self.minority_indices)))
        oversampled_minority_ = np.random.choice(
            self.minority_indices,
            self.minority_count - len(oversampled_minority),
            replace=False,
        )
        combined_indices = np.concatenate([self.majority_indices, oversampled_minority, oversampled_minority_])

        if len(combined_indices) > self.total_size:
            combined_indices = np.random.choice(combined_indices, self.total_size, replace=False)
        else:
            combined_indices = np.tile(combined_indices, int(self.total_size / len(combined_indices)))
            combined_indices_ = np.random.choice(
                combined_indices,
                self.total_size - len(combined_indices),
                replace=False,
            )
            combined_indices = np.concatenate([combined_indices, combined_indices_])
            np.random.shuffle(combined_indices)

        return iter(combined_indices)

    def __len__(self):
        return self.total_size


class CustomCollator:
    def __init__(self, tokenizer, max_seq_len=128, max_content_len=100):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_content_len = max_content_len

    def __call__(self, batch):
        sequences_, labels = zip(*batch)

        # Truncate each log sequence before flattening.
        sequences = [seq[: self.max_seq_len] for seq in sequences_]

        data, seq_positions = merge_data(sequences)
        # Drop the first 0; downstream code uses these values as split boundaries.
        seq_positions = seq_positions[1:]

        # Tokenize the flattened list of log messages.
        inputs = self.tokenizer(
            data,
            return_tensors="pt",
            max_length=self.max_content_len,
            padding=True,
            truncation=True,
        )

        labels = np.array(labels).astype(object)
        labels[labels == 0] = "normal"
        labels[labels == 1] = "anomalous"

        return {
            "inputs": inputs,
            "seq_positions": torch.tensor(seq_positions, dtype=torch.long),
            "labels": labels,
        }
