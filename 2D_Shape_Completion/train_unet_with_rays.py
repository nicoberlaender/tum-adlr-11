import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm

from types import SimpleNamespace
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split


from model.my_unet_model import UNet
from dataset.create_ray_dataset import ImageDatasetWithRays



def main():

  # Define hyperparameters
    hyperparameters = {
        "epochs": 1,
        "batch_size": 16,
        "learning_rate": 1e-3
}

    # Control whether to use wandb
    use_wandb = True

    if use_wandb:
        # Initialize wandb with project configuration
        wandb.init(project='unet-training', name='shape_compleetion', config=hyperparameters)
        config = wandb.config  # Directly use wandb.config
    else:
        # Create a SimpleNamespace for offline configuration
        config = SimpleNamespace(**hyperparameters)

    # Define the device (GPU or mps or cpu)
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'

    # Step 3: Reload Dataset and DataLoader with the Updated Transform
    dataset = ImageDatasetWithRays('data_with_rays', num_samples=100, transform=transforms.Compose([ToTensor()]))

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # Remaining 20% for validation

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders for both training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    print(len(train_loader))

    model = UNet(1, 16, 1)
    model= model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    global_step = 0

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        num_train_batches = 0

        # Training phase
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs} [Train]')
        for batch in train_pbar:
            # Unpack the batch
            images, targets = batch
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()  # Just add the loss
            num_train_batches += 1

            if global_step % 10 == 0:
                wandb.log({
                "batch/train_loss": loss.item(),
                "epoch": epoch,
                "batch": global_step,
            })

            global_step += 1

            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss /= num_train_batches

        # Validation phase
        model.eval()
        val_loss = 0.0
        num_val_batches = 0

        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config.epochs} [Valid]')
        with torch.no_grad():
            for batch in val_pbar:
                # Unpack the batch
                images, targets = batch
                # Move tensors to the device
                images = images.to(device)
                targets = targets.to(device)

                output = model(images) 
                # Compute loss
                loss =criterion(output, targets)
                val_loss += loss.item()
                num_val_batches += 1

                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        val_loss /= num_val_batches

        # Log metrics to wandb
        if (use_wandb):
            wandb.log({ 
            "epoch/train_loss": train_loss,
            "epoch/val_loss": val_loss,
            "epoch": epoch
        })

        # Print loss for each epoch
        print(f"Epoch {epoch+1}/{config.epochs}, "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Finish wandb logging
    if (use_wandb):
        wandb.finish()

    # Save the model
    torch.save(model.state_dict(), "model_weights_with_rays.pth")
    torch.save(model, "model_full_with_rays.pth")

if __name__ == '__main__':
    main()
