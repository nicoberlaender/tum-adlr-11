import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from types import SimpleNamespace
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import ConcatDataset

from model.my_unet_model import UNet
from dataset.create_dataset_file import ImageDataset



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
        wandb.init(project='unet-training', name='shape_compleetion_28.12', config=hyperparameters)
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
    dataset = ImageDataset('2D_Shape_Completion/data', num_samples=400, len_dataset=2500)

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

    model = UNet(1, 32, 1)
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
    best_val_loss = float('inf')  # Initialize to a large number

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
                loss =criterion(output, targets)
                val_loss += loss.item()
                num_val_batches += 1

                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        val_loss /= num_val_batches

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch' : epoch,
                'num_train_batches' :num_train_batches,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                }
                        
            # Ensure the directory exists
            os.makedirs('saved_models', exist_ok=True)
            torch.save(checkpoint,'saved_models/best_model.pth')

        # Log metrics to wandb
        if (use_wandb == True):
            wandb.log({ 
            "epoch/train_loss": train_loss,
            "epoch/val_loss": val_loss,
            "epoch": epoch
        })

        # Print loss for each epoch
        print(f"Epoch {epoch+1}/{config.epochs}, "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Finish wandb logging
    if (use_wandb ==True):
        wandb.finish()

    # Save the model
    torch.save(model.state_dict(), "model_weights.pth")
    torch.save(model, "model_full.pth")





#CODE TO TRAIN TO EVALUATE MORE OFTEN !!!!
def train_and_val():

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
        wandb.init(project='unet-training', name='shape_compleetion_28.12', config=hyperparameters)
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
    dataset = ImageDataset('2D_Shape_Completion/data', num_samples=400, len_dataset=2500)

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

    model = UNet(1, 32, 1)
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
    best_val_loss = float('inf')  # Initialize to a large number

    # Calculate the number of repeats needed to match the length of the training set
    num_repeats = (len(train_loader) + len(val_loader) - 1) // len(val_loader)  # Round up

    # Extend the validation dataset by duplicating it
    extended_val_dataset = ConcatDataset([val_dataset] * num_repeats)

    # Create a DataLoader for the extended validation dataset
    extended_val_loader = DataLoader(extended_val_dataset, batch_size=config.batch_size, shuffle=False)

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        num_train_batches = 0
        val_loss = 0.0
        num_val_batches = 0

        # Creazione della barra di progresso unica
        train_and_val_pbar = tqdm(zip(train_loader, extended_val_loader), 
                                total=len(train_loader),
                                desc=f"Epoch {epoch+1}/{config.epochs}")

        for train_batch, val_batch in train_and_val_pbar:
            # *** Training Phase ***
            # Unpack the training batch
            train_images, train_targets = train_batch
            train_images = train_images.to(device)
            train_targets = train_targets.to(device)

            # Forward pass
            optimizer.zero_grad()
            train_output = model(train_images)
            train_batch_loss = criterion(train_output, train_targets)

            # Backward pass and optimization
            train_batch_loss.backward()
            optimizer.step()
            scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']

            # Accumulate training loss
            train_loss += train_batch_loss.item()
            num_train_batches += 1

            # *** Validation Phase ***
            model.eval()
            with torch.no_grad():
                # Unpack the validation batch
                val_images, val_targets = val_batch
                val_images = val_images.to(device)
                val_targets = val_targets.to(device)

                # Forward pass
                val_output = model(val_images)
                val_batch_loss = criterion(val_output, val_targets)

                # Accumulate validation loss
                val_loss += val_batch_loss.item()
                num_val_batches += 1

            # Update the tqdm bar with both training and validation losses
            train_and_val_pbar.set_postfix({
                'Train Loss': f'{train_batch_loss.item():.4f}',
                'Val Loss': f'{val_batch_loss.item():.4f}',
                'LR': f'{scheduler.get_last_lr()[0]:.6f}'
            })


            if global_step % 10 == 0 and use_wandb:
                wandb.log({
                "batch/train_loss": train_batch_loss.item(),
                "batch/val_loss": val_batch_loss.item(),
                "batch/learning_rate": current_lr,
                "epoch": epoch,
                "batch": global_step,
            })

            # Save the best model
        if val_batch_loss.item() < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model_state_dict': model.state_dict()}, "saved_models/best_model.pth")

            global_step +=1

        # Compute average losses for the epoch
        train_loss /= num_train_batches
        val_loss /= num_val_batches

        

        

        print(f"Epoch {epoch+1}/{config.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


    # Finish wandb logging
    if (use_wandb ==True):
        wandb.finish()

    # Save the model
    torch.save(model.state_dict(), "model_weights.pth")
    torch.save(model, "model_full.pth")


    
if __name__ == '__main__':
    # Prompt the user for input
    user_input = input("Enter 1 to run the main function or 2 to run the train and validation function: ")

    # Call the corresponding function based on user input
    if user_input == '1':
        main()  # Calls the main function
    elif user_input == '2':
        train_and_val()  # Calls the train_and_val function
    else:
        print("Invalid input. Please enter 1 or 2.")