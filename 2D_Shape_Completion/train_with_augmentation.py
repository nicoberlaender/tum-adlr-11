import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm

from types import SimpleNamespace
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import OneCycleLR

from model.my_unet_model import UNet
from dataset.image_dataset import ImageDataset

def main():

  # Define hyperparameters
    hyperparameters = {
        "epochs": 100,
        "batch_size": 64,
        "learning_rate": 1e-3
}

    # Control whether to use wandb
    use_wandb = True

    if use_wandb:
        # Initialize wandb with project configuration
        wandb.init(project='unet-training', name='shape_compleetion_augmentation', config=hyperparameters)
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

    dataset = ImageDataset('data')

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # Remaining 20% for validation

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(1.0, 1.0)
        ),
    ])

    train_dataset.dataset.transform = train_transform


    # Create DataLoaders for both training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    total_steps = len(train_loader) * config.epochs

    model = UNet(1, 16, 1)
    model= model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=0.3,  # Spend 30% of iterations in warmup
        anneal_strategy='cos'
    )

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
            scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']

            train_loss += loss.item()  # Just add the loss
            num_train_batches += 1

            if global_step % 10 == 0 and use_wandb:
                wandb.log({
                "batch/train_loss": loss.item(),
                "batch/learning_rate": current_lr,
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
                loss = criterion(output, targets)
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
    torch.save(model, "model_augmentation.pth")
    
if __name__ == '__main__':
    main()