import os
from typing import Dict

import torch
from loguru import logger
from torch.utils.data import DataLoader

from model import GPTDecoder


class TrainingLoop:
    def __init__(self, model_params: Dict, training_params: Dict) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
        self._initialise_training_params(training_params)
        self.model = GPTDecoder(model_params=model_params).to(self.device)  # Move model to the device
        self.num_params = sum(param.numel() for param in self.model.parameters())
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.best_val_loss = float('inf')  # To track the best validation loss
        self.start_epoch = 0  # To track the starting epoch (useful for resuming training)

    def _initialise_training_params(self, training_params: Dict):
        self.lr = training_params.get('learning_rate', 3e-4)
        self.epochs = training_params.get('num_epochs', 5000)
        self.eval_interval = training_params.get('eval_interval', 500)
        self.eval_iters = training_params.get('eval_iters', 500)

    @torch.no_grad()
    def estimate_loss(self, train_loader, val_loader):
        """
        Estimates loss on train and val sets based on eval_iters random batches.
        """
        all_losses = {}
        self.model.eval()

        for split, loader in zip(['train', 'val'], [train_loader, val_loader]):
            losses = []
            loader_iter = iter(loader)  # Create an iterator for the DataLoader
            for _ in range(self.eval_iters):
                try:
                    Xb, Yb, mask = next(loader_iter)  # Get the next batch
                except StopIteration:
                    loader_iter = iter(loader)  # Restart the iterator if it runs out of data
                    Xb, Yb, mask = next(loader_iter)

                # Move data to the same device as the model
                Xb, Yb, mask = Xb.to(self.device), Yb.to(self.device), mask.to(self.device)

                _, loss = self.model(Xb, Yb, mask)
                losses.append(loss.item())
            
            all_losses[split] = sum(losses) / len(losses)  # Mean loss over eval_iters batches

        self.model.train()
        return all_losses

    def save_checkpoint(self, epoch: int, save_path: str, is_best: bool = False):
        """
        Saves the model and optimizer state dictionaries.

        Args:
            epoch: Current epoch number.
            save_path: Path to save the model.
            is_best: Whether this is the best model based on validation loss.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        save_file = f"{save_path}/checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, save_file)
        logger.info(f"Checkpoint saved: {save_file}")

        if is_best:
            best_file = f"{save_path}/best_model.pth"
            torch.save(checkpoint, best_file)
            logger.info(f"Best model saved: {best_file}")

    def load_checkpoint(self, load_path: str):
        """
        Loads the model and optimizer state dictionaries from a checkpoint.

        Args:
            load_path: Path to the checkpoint file.
        """
        checkpoint = torch.load(load_path, map_location=self.device)  # Ensure checkpoint is loaded to the correct device
        self.model.load_state_dict(checkpoint['model_state_dict'])  # Load model weights
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Load optimizer state
        self.best_val_loss = checkpoint['best_val_loss']  # Restore best validation loss
        # self.start_epoch = checkpoint['epoch'] + 1  # Resume from the next epoch
        logger.info(f"Checkpoint loaded from: {load_path}, resuming from epoch {self.start_epoch}")

    def train(self, train_loader: DataLoader, val_loader: DataLoader, save_model_path: str, resume_path: str = None):
        """
        Training loop with optional resume functionality.

        Args:
            train_loader: DataLoader for the training dataset.
            val_loader: DataLoader for the validation dataset.
            save_model_path: Directory to save checkpoints.
            resume_path: Path to a checkpoint to resume training from.
        """
        os.makedirs(save_model_path, exist_ok=True)

        # Load checkpoint if resume_path is provided
        if resume_path:
            self.load_checkpoint(resume_path)

        logger.debug(f"Number of model parameters: {self.num_params}")
        train_iter = iter(train_loader)

        for epoch in range(self.start_epoch, self.epochs):
            try:
                Xb, Yb, mask = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                Xb, Yb, mask = next(train_iter)

            # Move data to the same device as the model
            Xb, Yb, mask = Xb.to(self.device), Yb.to(self.device), mask.to(self.device)

            _, loss = self.model(Xb, Yb, mask)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            # Evaluate and save model periodically
            if epoch % self.eval_interval == 0:
                losses = self.estimate_loss(train_loader, val_loader)
                logger.info(f"For epoch {epoch}: Train loss-> {losses['train']} | Val loss-> {losses['val']}")

                # Save model if validation loss improves
                if losses['val'] < self.best_val_loss:
                    self.best_val_loss = losses['val']
                    self.save_checkpoint(epoch=epoch, save_path=save_model_path, is_best=True)
                else:
                    self.save_checkpoint(epoch=epoch, save_path=save_model_path, is_best=False)
