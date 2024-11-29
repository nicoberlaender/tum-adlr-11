import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt

from types import SimpleNamespace
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split


from model.simple_unet import UNet
from dataset.create_dataset_file import ImageDataset



def main():

  # Define hyperparameters
    hyperparameters = {
        "epochs": 250,
        "batch_size": 10,
        "learning_rate": 1e-3
}

    # Control whether to use wandb
    use_wandb = False

    if use_wandb:
        # Initialize wandb with project configuration
        wandb.init(project='unet-training', name='overfitting', config=hyperparameters)
        config = wandb.config  # Directly use wandb.config
    else:
        # Create a SimpleNamespace for offline configuration
        config = SimpleNamespace(**hyperparameters)

    # Define the device (GPU or mps or cpu)
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    if torch.backends.mps.is_available():
        device = 'mps'

    # Step 3: Reload Dataset and DataLoader with the Updated Transform
    dataset = ImageDataset('overfit_data', len_dataset=1)

    # Create DataLoaders for both training and validation sets
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    model = UNet(1, 16, 1)
    model= model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0

        # Training phase
        for batch in train_loader:
            # Unpack the batch
            images, targets = batch

            # Move tensors to the device
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            optimizer.zero_grad()
            output = model(images)
            #output = torch.sigmoid(output)
           
            # Compute loss
            loss = criterion(output, targets)
            #print(loss)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        # Log metrics to wandb
        if (use_wandb == True):
            wandb.log({ 
            "train_loss": train_loss,
            "epoch": epoch
        })

        # Print loss for each epoch
        print(f"Epoch {epoch+1}/{config.epochs}, "
              f"Train Loss: {train_loss:.4f}")

    # Finish wandb logging
    if (use_wandb ==True):
        wandb.finish()

    # Save the model
    torch.save(model.state_dict(), "model_weights_overfitting.pth")
    torch.save(model, "model_full_overfitting.pth")

if __name__ == '__main__':
    main()
