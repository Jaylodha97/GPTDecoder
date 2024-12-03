from typing import Dict, List

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset


class CustomTextDataset(Dataset):
    def __init__(self, df, text_col: int, label_col: int, labels_lookup_dict: Dict[str, int] = None, block_size: int = None):
        """
        Custom Dataset for text data. Uses tiktoken for tokenization.
        
        Args:
            df: DataFrame containing the dataset.
            text_col: Column index for text input.
            label_col: Column index for labels.
            labels_lookup_dict: Predefined label-to-index mapping (optional).
            block_size: Maximum sequence length (optional). Determined if not provided.
        """
        self.texts = df[text_col].values.tolist()  # Extract text column
        self.labels = df[label_col].values.tolist()  # Extract label column
        
        # Create label lookup dictionary if not provided
        if labels_lookup_dict is None:
            unique_labels = sorted(set(self.labels))
            self.labels_lookup_dict = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.labels_lookup_dict = labels_lookup_dict

        # Reverse mapping for convenience
        self.reverse_labels_lookup_dict = {v: k for k, v in self.labels_lookup_dict.items()}
        
        # Convert labels to integer indices
        self.labels = [self.labels_lookup_dict[label] for label in self.labels]
        
        # Tokenize text inputs
        self.tokenizer = tiktoken.get_encoding("cl100k_base") # hardcoding this for now
        self.vocab_size = self.tokenizer.n_vocab  # Vocabulary size from the tokenizer
        self.encoded_texts = [self.encode_text(text) for text in self.texts]
        
        # Determine block size if not provided
        self.block_size = block_size if block_size else self.determine_block_size(self.encoded_texts)

    def encode_text(self, text: str) -> List[int]:
        """Tokenizes input text into a list of integers."""
        return self.tokenizer.encode(str(text))
    
    def determine_block_size(self, encoded_texts: List[List[int]]) -> int:
        """Determines the maximum sequence length for padding/truncation."""
        return max(len(tokens) for tokens in encoded_texts)

    def __len__(self):
        return len(self.encoded_texts)

    def __getitem__(self, idx):
        """Prepares a single sample with padding and mask."""
        tokens = self.encoded_texts[idx]
        label = self.labels[idx]

        # Pad or truncate tokens to block_size
        if len(tokens) < self.block_size:
            tokens = tokens + [0] * (self.block_size - len(tokens))  # Pad with zeros
        else:
            tokens = tokens[:self.block_size]

        # Convert to tensors
        tokens = torch.tensor(tokens, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)
        mask = (tokens != 0).float()  # Mask for padding

        return tokens, label, mask


class CustomDataLoader:
    def __init__(self, train_df, val_df, text_col: int, label_col: int, batch_size: int):
        """
        Custom DataLoader for train and validation datasets.
        
        Args:
            train_df: DataFrame for training data.
            val_df: DataFrame for validation data.
            text_col: Column index for text input.
            label_col: Column index for labels.
            batch_size: Batch size for loading data.
        """
        self.batch_size = batch_size
        
        # Prepare datasets
        self.train_dataset = CustomTextDataset(train_df, text_col, label_col)
        self.val_dataset = CustomTextDataset(val_df, text_col, label_col, 
                                             labels_lookup_dict=self.train_dataset.labels_lookup_dict, 
                                             block_size=self.train_dataset.block_size)

    
    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to merge a list of samples into a batch.
        """
        tokens, labels, masks = zip(*batch)
        tokens = torch.stack(tokens)
        labels = torch.stack(labels)
        masks = torch.stack(masks)
        return tokens, labels, masks

    def get_train_loader(self, shuffle: bool = True):
        """
        Returns a DataLoader for the training dataset containing a batch of input_tokens, labels and corresponding mask tensor.

        Args:
            shuffle (bool): Whether to shuffle the data at each epoch. Defaults to True.

        Returns:
            DataLoader: DataLoader for the training dataset.
        """
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=shuffle, collate_fn=self.collate_fn)
        return self.train_loader

    def get_val_loader(self, shuffle: bool = False):
        """
        Returns a DataLoader for the validation dataset containing a batch of input_tokens, labels and corresponding mask tensor.

        Args:
            shuffle (bool): Whether to shuffle the data at each epoch. Defaults to False.

        Returns:
            DataLoader: DataLoader for the validation dataset.
        """
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=shuffle, collate_fn=self.collate_fn)
        return self.val_loader
