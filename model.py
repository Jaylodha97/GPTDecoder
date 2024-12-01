from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Implements scaled dot-product self-attention.

    Attributes:
        head_size (int): Size of each attention head.
        key (nn.Linear): Linear layer to compute the Key matrix.
        query (nn.Linear): Linear layer to compute the Query matrix.
        value (nn.Linear): Linear layer to compute the Value matrix.
        dropout (nn.Dropout): Dropout layer for regularization.
    """
    def __init__(self, head_size: int, n_embds: int, dropout: float) -> None:
        """
        Initializes the SelfAttention module.

        Args:
            head_size (int): Size of each attention head.
            n_embds (int): Dimensionality of the input embeddings.
            dropout (float): Dropout probability for regularization.
        """
        super().__init__()

        self.head_size = head_size
        self.key = nn.Linear(n_embds, head_size, bias=False)
        self.query = nn.Linear(n_embds, head_size, bias=False)
        self.value = nn.Linear(n_embds, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None) -> torch.Tensor:
        """
        Computes the self-attention output for the given input.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, n_embds), where
                B = batch size,
                T = sequence length,
                n_embds = embedding dimension.
            mask (torch.Tensor, optional): Attention mask of shape (B, T) to ignore certain positions. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (B, T, head_size).
        """
        B, T, n_embds = x.shape

        # Compute Q, K, V matrices
        q = self.query(x)  # (B,T,head_size)
        k = self.key(x)    # (B,T,head_size)
        v = self.value(x)  # (B,T,head_size)

        # Compute scaled dot-product attention
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5  # (B,T,T)

        # Apply attention mask to ignore padding tokens
        if mask is not None:
            mask = mask.unsqueeze(1)  # (B, 1, T) to broadcast along attention weights
            wei = wei.masked_fill(mask == 0, float('-inf'))  # Set -inf where mask is 0 (padding)

        wei = F.softmax(wei, dim=-1)  # Softmax on the last dimension (attention scores)
        wei = self.dropout(wei)       # Apply dropout

        # Weighted sum of values
        out = wei @ v  # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)
        return out


class MultiHeadSelfAttention(nn.Module):
    """
    Implements multi-head self-attention, where multiple attention heads
    operate independently and their outputs are concatenated and projected back.

    Attributes:
        attn_heads (nn.ModuleList): List of `SelfAttention` modules, one for each head.
        proj (nn.Linear): Linear projection layer to map concatenated outputs back to the input dimensionality.
        dropout (nn.Dropout): Dropout layer for regularization.
    """

    def __init__(self, num_heads: int, head_size: int, dropout: float) -> None:
        """
        Initializes the MultiHeadSelfAttention module.

        Args:
            num_heads (int): Number of attention heads.
            head_size (int): Size of each attention head.
            dropout (float): Dropout probability for regularization.
        """
        super().__init__()
        self.attn_heads = nn.ModuleList([SelfAttention(head_size, head_size * num_heads, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, head_size * num_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Computes the output of the multi-head self-attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, n_embds), where
                B = batch size,
                T = sequence length,
                n_embds = embedding dimension.
            mask (torch.Tensor, optional): Attention mask of shape (B, T) to ignore certain positions. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (B, T, n_embds).
        """
        # Each head processes the input independently and then we concatenate
        multi_head_out = torch.cat([head(x, mask) for head in self.attn_heads], dim=-1)  # (B, T, n_embds)
        out = self.dropout(self.proj(multi_head_out))  # Project concatenated heads back to n_embds
        return out


class FeedForwardLayer(nn.Module):
    """
    Implements a feed-forward neural network layer with two linear transformations
    and a ReLU activation in between. Includes dropout for regularization.

    Attributes:
        ffw (nn.Sequential): A sequential container of the feed-forward network components:
            - First linear layer: Expands dimensionality by a factor of 4.
            - ReLU activation: Applies non-linearity.
            - Second linear layer: Reduces dimensionality back to the original size.
            - Dropout: Adds regularization to prevent overfitting.
    """

    def __init__(self, n_embds: int, dropout: float) -> None:
        """
        Initializes the FeedForwardLayer module.

        Args:
            n_embds (int): Dimensionality of the input embeddings.
            dropout (float): Dropout probability for regularization.
        """
        super().__init__()
        self.ffw = nn.Sequential(
            nn.Linear(n_embds, n_embds * 4),
            nn.ReLU(),
            nn.Linear(n_embds * 4, n_embds),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the feed-forward network.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, n_embds), where
                B = batch size,
                T = sequence length,
                n_embds = embedding dimension.

        Returns:
            torch.Tensor: Output tensor of shape (B, T, n_embds), with the same shape as input.
        """
        out = self.ffw(x)
        return out


class Block(nn.Module):
    """
    Represents a single transformer block consisting of:
    - Multi-head self-attention mechanism.
    - Feed-forward neural network.
    - Layer normalization applied before each sub-layer.
    - Residual connections after each sub-layer.

    Attributes:
        multi_headed_attn (MultiHeadSelfAttention): Multi-head self-attention mechanism.
        ffw_layer (FeedForwardLayer): Feed-forward network.
        ln1 (nn.LayerNorm): Layer normalization before the multi-head attention.
        ln2 (nn.LayerNorm): Layer normalization before the feed-forward layer.
    """
    def __init__(self, n_embds: int, num_heads: int, dropout: float) -> None:
        """
        Initializes the MultiHeadSelfAttention module.

        Args:
            num_heads (int): Number of attention heads.
            head_size (int): Size of each attention head.
            dropout (float): Dropout probability for regularization.
        """
        super().__init__()
        head_size = n_embds // num_heads
        self.multi_headed_attn = MultiHeadSelfAttention(num_heads, head_size, dropout)
        self.ffw_layer = FeedForwardLayer(n_embds, dropout)
        self.ln1 = nn.LayerNorm(n_embds)
        self.ln2 = nn.LayerNorm(n_embds)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Performs a forward pass through the transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, n_embds), where
                B = batch size,
                T = sequence length,
                n_embds = embedding dimension.
            mask (torch.Tensor, optional): Attention mask of shape (B, T) to ignore certain positions. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (B, T, n_embds), with the same shape as the input.
        """
        # Apply multi-head attention and add residual connection
        x = x + self.multi_headed_attn(self.ln1(x), mask)
        # Apply feed-forward network and add residual connection
        x = x + self.ffw_layer(self.ln2(x))
        return x


class GPTDecoder(nn.Module):
    """
    Implements a GPT-style decoder for sequence classification tasks.

    The decoder uses:
    - Token embeddings and positional embeddings.
    - Multiple transformer blocks for processing.
    - Final layer normalization and a classification head.

    Attributes:
        token_embeddings (nn.Embedding): Embedding layer for tokens.
        positional_embeddings (nn.Embedding): Embedding layer for positional encodings.
        blocks (nn.ModuleList): List of transformer blocks for processing.
        layernorm (nn.LayerNorm): Final layer normalization.
        classifier (nn.Linear): Linear layer for classification.
    """

    def __init__(self, model_params: Dict) -> None:
        """
        Initializes the GPTDecoder module with the given model parameters.

        Args:
            model_params (Dict): Dictionary containing model hyperparameters:
                - "vocab_size" (int): Vocabulary size for token embeddings.
                - "num_embeddings" (int): Dimensionality of embeddings.
                - "block_size" (int): Maximum sequence length.
                - "num_heads" (int): Number of attention heads.
                - "num_layers" (int): Number of transformer blocks.
                - "output_classes" (int): Number of output classes.
                - "dropout" (float): Dropout probability.
                - "device" (str): Device to run the model on ("cpu" or "cuda").
        """
        super().__init__()

        self._initialise_model_params(model_params)
        
        self.token_embeddings = nn.Embedding(self.vocab_size, self.n_embds)
        self.positional_embeddings = nn.Embedding(self.block_size, self.n_embds)
        self.blocks = nn.ModuleList([Block(self.n_embds, self.n_heads, self.dropout) for _ in range(self.n_layers)])
        self.layernorm = nn.LayerNorm(self.n_embds)
        self.classifier = nn.Linear(self.n_embds, self.num_classes)

    def _initialise_model_params(self, model_params: Dict) -> None:
        """
        Initializes model parameters from a dictionary.

        Args:
            model_params (Dict): Dictionary of model hyperparameters.
        """

        self.vocab_size = model_params.get("vocab_size")
        self.n_embds = model_params.get("num_embeddings")
        self.block_size = model_params.get("block_size")
        self.n_heads = model_params.get("num_heads")
        self.n_layers = model_params.get("num_layers")
        self.num_classes = model_params.get("output_classes")
        self.dropout = model_params.get("dropout")
        self.device = model_params.get("device")

    def forward(self, idx: torch.tensor, targets: torch.tensor=None, mask: torch.tensor = None) -> Tuple[torch.tensor, float]:
        """
        Performs a forward pass through the GPTDecoder.

        Args:
            idx (torch.Tensor): Input token indices of shape (B, T), where
                B = batch size,
                T = sequence length.
            targets (torch.Tensor, optional): Target labels of shape (B). Defaults to None.
            mask (torch.Tensor, optional): Attention mask of shape (B, T). Defaults to None.

        Returns:
            Tuple[torch.Tensor, float]: A tuple containing:
                - logits (torch.Tensor): Output logits of shape (B, num_classes).
                - loss (float): Cross-entropy loss if targets are provided; otherwise, None.
        """

        B, T = idx.shape
        token_embeddings = self.token_embeddings(idx)  # B, T -> B, T, n_embds
        positional_embeddings = self.positional_embeddings(torch.arange(T, device = self.device))  # T, n_embds

        # Combine token and positional embeddings
        x = token_embeddings + positional_embeddings  # B, T, n_embds

        # Forward pass through each Block, passing the mask
        for block in self.blocks:
            x = block(x, mask)  # Pass mask to each block individually

        layer_norm_out = self.layernorm(x)  # (B, T, n_embds)

        # Pooling to get a single vector per sequence for classification
        pooled_output = layer_norm_out.mean(dim=1)  # Average pooling across T dimension

        logits = self.classifier(pooled_output)  # B, n_embds -> B, num_classes

        # Calculate loss if targets are provided
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits, targets)  # Cross-entropy loss for classification

        return logits, loss
