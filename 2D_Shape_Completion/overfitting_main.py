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


from model.my_unet_model import UNet
from transforms.mean_std import calculate_mean_std
from loss.losses import FocalLoss, WeightedBCE, DiceLoss, JaccardLoss
from dataset.create_dataset_file import ImageDataset



def main():

  # Define hyperparameters
    hyperparameters = {
        "epochs": 3000,
        "batch_size": 16,
        "learning_rate": 1e-4, 
        "len_dataset": 2, 
        "len_train":1,         
        }
    #hyperparameters["loss"] = "WeightedBCE"
    hyperparameters["loss"] = "WeightedBCE"
    # Control whether to use wandb
    use_wandb = True


    if use_wandb:
        # Initialize wandb with project configuration
        wandb.init(project='unet-training', name='overfitting', config=hyperparameters)
        config = wandb.config  # Directly use wandb.config
    else:
        # Create a SimpleNamespace for offline configuration
        config = SimpleNamespace(**hyperparameters)

    # Define the device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Esempio di come usare il dataset
    root_dir = "C:/Users/iacop/Desktop/Programmazione/Github/tum-adlr-11/data"  # Cartella contenente le sottocartelle numerate
    transform = ToTensor()  # Transform images to tensors

   # Apply the transformation: convert to tensor and change dtype to float32
    transform = transforms.Compose([
    transforms.ToTensor(),  # Converts to a tensor and scales the values to [0, 1]
    ])

    # Step 3: Reload Dataset and DataLoader with the Updated Transform
    normalized_dataset = ImageDataset(root_dir=root_dir, num_samples=50, len_dataset=config.len_dataset, transform=transform)


    train_dataset, val_dataset = random_split(normalized_dataset, [config.len_train, len(normalized_dataset)-config.len_train])

    # Create DataLoaders for both training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = UNet(1, 4, 1, sigmoid=False)
    model= model.to(device)

    if (config.loss == "FocalLoss"):
        criterion = FocalLoss(alpha=0.25, gamma=2)
    elif (config.loss == "WeightedBCE"):
        criterion = WeightedBCE([0.1, 2])
    elif (config.loss == "Jaccard"):
        criterion = JaccardLoss()

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
           
            # Compute loss
            targets = targets.unsqueeze(1)
 
            loss = criterion(output, targets)
            #print(loss)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            print(loss.item())
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        i=0
        with torch.no_grad():
            for batch in val_loader:
                # Unpack the batch
                images, targets = batch
                # Move tensors to the device
                images = images.to(device)
                targets = targets.to(device)

                output = model(images) 
                # Compute loss
                targets = targets.unsqueeze(1)
                loss =criterion(output, targets)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)

        # Log metrics to wandb
        if (use_wandb == True):
            wandb.log({ 
            "train_loss": train_loss,
            "val_loss": val_loss,
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

if __name__ == '__main__':
    main()
