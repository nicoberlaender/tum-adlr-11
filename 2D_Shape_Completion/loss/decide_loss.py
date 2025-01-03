import torch
import torch.nn as nn



from loss.losses import FocalLoss




# Function to choose the loss function
def get_loss_function():
    print("Choose the loss function:")
    print("1: Normal BCE Loss")
    print("2: Weighted BCE Loss")
    print("3: Focal Loss")
    
    choice = input("Enter the number corresponding to your choice: ")
    if choice == '1':
        print("Using Normal BCE Loss.")
        return nn.BCEWithLogitsLoss()
    elif choice == '2':
        print("Using Weighted BCE Loss.")
        weights = torch.tensor([0.1, 1.0])  # Background and object weights
        return nn.BCEWithLogitsLoss(pos_weight=weights[1])
    elif choice == '3':
        print("Using Focal Loss.")
        return FocalLoss(alpha=1, gamma=2)
    else:
        print("Invalid choice. Defaulting to Normal BCE Loss.")
        return nn.BCEWithLogitsLoss()
