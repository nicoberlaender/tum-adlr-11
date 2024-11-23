from create_dataset_file import ImageDataset
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import wandb
from torchvision import transforms
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
from my_unet_model import UNet

# Calcolo media e deviazione standard
def calculate_mean_std(dataset, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    mean_sum = 0
    std_sum = 0
    total_pixels = 0

    for inputs, _ in loader:
        inputs = inputs.view(inputs.size(0), -1)  # Flatten images (batch_size, height * width)
        mean_sum += inputs.mean(1).sum().item()
        std_sum += inputs.std(1).sum().item()
        total_pixels += inputs.size(0)

    mean = mean_sum / total_pixels
    std = std_sum / total_pixels
    return mean, std


# Dataset originale senza normalizzazione
root_dir = "C:/Users/iacop/Desktop/Programmazione/Github/tum-adlr-11/data"
transform = transforms.ToTensor()
dataset = ImageDataset(root_dir=root_dir, num_samples=10, transform=transform)

# Calcola mean e std
mean, std = calculate_mean_std(dataset)
print(f"Dataset mean: {mean}, std: {std}")

# Aggiunge la normalizzazione direttamente al dataset esistente
dataset.transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[mean], std=[std])
])


# Divide il dataset in training e validation set
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Define the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the UNet model
model = UNet(1, 4, 1).to(device)  # Update output channels to match the number of classes

# Use CrossEntropyLoss for multi-class segmentation
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

# Training and Validation Loop
for epoch in range(wandb.config.epochs):
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        images, targets = batch

        # Move tensors to device
        images, targets = images.to(device), targets.to(device)

        print(images.shape(),targets.shape())

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, targets.long())  # Ensure targets are of type long

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_loader.dataset)

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            images, targets = batch
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets.long())
            val_loss += loss.item() * images.size(0)

    val_loss /= len(val_loader.dataset)

    # Log metrics
    wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})

    print(f"Epoch {epoch + 1}/{wandb.config.epochs}, "
          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Save model weights
torch.save(model.state_dict(), "model_weights.pth")

# Save full model
torch.save(model, "model_full.pth")

wandb.finish()
