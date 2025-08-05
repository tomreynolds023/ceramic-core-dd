'''
Hyperparameter tuning script for the ResNet18-based ceramic core defect detection model

Uses Ray Tune for distributed hyperparameter tuning and training.
'''

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
import os
from pathlib import Path
import tempfile

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchinfo import summary

import torchvision

import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms.v2 as transforms
import torchvision.models as models

from ray import tune
from ray import train
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle

from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

plt.rcParams['savefig.dpi'] = 300 # Enables high resolution figures


def load_data(data_dir="Core1_Fulls/Core1_splits_overlap/"):
    '''
    Load the dataset from the specified directory and apply transformations.

    Parameters:
        data_dir (str): Path to the dataset directory.
    
    Returns:
        train_dataset (torch.utils.data.Dataset): Training dataset. 
        val_dataset (torch.utils.data.Dataset): Validation dataset.
        test_dataset (torch.utils.data.Dataset): Test dataset.
    '''

    # Mean and standard deviation for normalization
    # These values are calculated from the training set of the dataset
    mean = (0.6001, 0.5423, 0.5025)
    std = (0.1197, 0.1146, 0.1126)

    data_transforms = transforms.Compose([transforms.ToImage(), 
                                        transforms.ToDtype(torch.float32, scale=True),
                                        transforms.Normalize(mean, std)
                                        ])
    

    # Load the dataset using ImageFolder
    dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
    test_length = int(len(dataset)/10)
    val_length = int(len(dataset)/10)
    train_length = len(dataset) - test_length - val_length 
    
    # Randomly split data into datasets - 80:10:10 train val test split
    # Generator means the split will be the same for every set of hyperparameters
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_length, test_length, val_length], generator=torch.Generator().manual_seed(42))

    return train_dataset, val_dataset, test_dataset


def train_model(config, data_dir=None, max_epochs=20): 
    '''
    Train the model with the given hyperparameters.
    
    Parameters:
        config (dict): Search space for hyperparameters.
        data_dir (str): Path to the dataset directory.
        max_epochs (int): Maximum number of training epochs.

    Returns:
        None
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Loads the ResNet18 model with pretrained weights
    # Freezes the first two layers and trains the last two layers
    model = models.resnet18(weights="IMAGENET1K_V1")
    num_ftrs = model.fc.in_features
    for param in model.parameters():
        param.requires_grad=False
    for param in model.layer3.parameters():
        param.requires_grad=True
    for param in model.layer4.parameters():
        param.requires_grad=True
        
    # Change the final layer to output 3 classes
    model.fc = nn.Linear(num_ftrs, 3)
    model.fc.requires_grad=True
    model = model.to(device)

    # Set up the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9) # Learning rate taken from config

    # Allow for resuming training from a checkpoint
    checkpoint = tune.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            model.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    # Load the data, put into DataLoaders
    train_dataset, val_dataset, test_dataset = load_data(data_dir)

    train_loader = DataLoader(dataset=train_dataset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=4)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=val_dataset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=4)

    # Training loop
    for epoch in range(start_epoch, max_epochs):
        # Model in training mode
        model.train()
        train_loss = 0.0
        train_loss_epoch = 0.0
        # Predict labels for each batch, update weights
        # See model_training.ipynb for more details on the training loop
        for batch_index, (data, targets) in enumerate(train_loader):
            data = data.to(device)
            targets = targets.to(device)
            scores = model(data)
            loss = criterion(scores, targets)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        # Calculate epoch training loss
        train_loss_epoch = train_loss / len(train_loader)
        
        # Print training loss every 10 epochs 
        if (epoch % 10 == 0):
            print(f'Train Loss after {epoch} epochs = {train_loss_epoch}')
        
        # Eval mode for validation loss
        # Stops the model updating gradients 
        model.eval()
        val_loss = 0.0
        val_loss_epoch = 0.0
        total = 0
        correct = 0
        with torch.no_grad():
            # again predict labels for each batch, but do not update weights
            for batch_index, (data, targets) in enumerate(val_loader):
                data = data.to(device)
                val_targets = targets.to(device)
                
                val_scores = model(data)
                _, predicted = torch.max(val_scores.data, 1)
                total += val_targets.size(0)
                correct += (predicted == val_targets).sum().item()
                
                loss = criterion(val_scores, val_targets)
                val_loss += loss.item()
    
        val_loss_epoch = val_loss / len(val_loader) # average validation loss
        if (epoch % 10 == 0):
            print(f'Val Loss after {epoch} epochs = {val_loss_epoch}')

        # Create checkpoint to allow pausing and resuming training
        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)

            # Report training and validation loss, and validation accuracy
            # This is used by Ray Tune to track the progress of the training
            train.report(
                {"train_loss": train_loss_epoch, "loss": val_loss_epoch, "accuracy": correct / total},
                checkpoint=checkpoint,
            )

    print("Finished Training")

def test_model(model, data_dir):
    '''
    Test the optimum trained model on the test dataset and compute metrics.

    Parameters:
        model (torch.nn.Module): The trained model to evaluate.
        data_dir (str): Path to the dataset directory.

    Returns:
        test_f1 (float): F1 score of the model on the test dataset.
    '''

    # Load the test dataset
    train_dataset, val_dataset, test_dataset = load_data(data_dir)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Lists to collect all predictions and true labels
    all_preds = []
    all_labels = []
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Make predictions on the test set
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)    
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics using sklearn
    test_accuracy = accuracy_score(all_labels, all_preds)
    test_recall_class = recall_score(all_labels, all_preds, average=None)
    test_recall = recall_score(all_labels, all_preds, average='macro')# Recall per class
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    cm_real = confusion_matrix(all_labels, all_preds)
    
    # Print metrics
    print(f"Best test accuracy                  = {test_accuracy:.4f}")
    print(f"Best test recall per class (macro)  = {test_recall_class:.4f}")
    print(f"Best test recall (macro)            = {test_recall:.4f}")
    print(f"Best test F1 score (macro)          = {test_f1:.4f}")
    print(f"Best confusion Matrix:\n{cm_real}")

    return test_f1

def main(num_samples=20, max_num_epochs=50):
    '''
    Main function to run the hyperparameter tuning and training process.
    
    Parameters:
        num_samples (int): Number of trials to run in the search space.
        max_num_epochs (int): Maximum number of epochs for training.
        
    Returns:
        None
    '''

    # Load data
    data_dir = os.path.abspath("Core1_Fulls/Core1_splits_overlap")
    load_data(data_dir)

    # Define the search space for hyperparameters
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([16, 32, 64, 128, 256]),
    }

    # Set up the scheduler for hyperparameter tuning
    # ASHAScheduler is used for early stopping of trials that are not performing well
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=5,
        reduction_factor=2,
    )

    # Run the hyperparameter tuning using Ray Tune
    result = tune.run(
        partial(train_model, data_dir=data_dir, max_epochs=max_num_epochs),
        resources_per_trial={"cpu": 2, "gpu": 0.25},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
    )

    # Print the best trial results
    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    # Load the best model and evaluate on the test set
    best_trained_model = models.resnet18()
    num_ftrs = best_trained_model.fc.in_features
    best_trained_model.fc = nn.Linear(num_ftrs, 3)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_trained_model.to(device)

    # Load the best checkpoint
    best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="loss", mode="min")
    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)

        best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])
        test_f1 = test_model(best_trained_model, data_dir, device)
        print("Best trial test set F1 score: {}".format(test_f1))

if __name__ == "__main__":
    # Run hyperparameter tuning with specified number of samples and epochs
    main(num_samples=30, max_num_epochs=20)