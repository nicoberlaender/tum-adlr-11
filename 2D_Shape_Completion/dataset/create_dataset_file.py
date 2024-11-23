import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from dataset.preprocessing import sample_pixels, segmap_to_binary, binary_to_image
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, root_dir, num_samples=10, transform=None):
        """
        Args:
            root_dir (string): Directory containing all the numbered folders with images.
            num_samples (int): Number of pixels to sample for input images.
            transform (callable, optional): Transformations to apply to the images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.num_samples = num_samples

        # Gather all image paths in the root directory
        for file in os.listdir(root_dir):
            image_path = os.path.join(root_dir, file)
            if os.path.isfile(image_path):  # Check if it's a file
                self.data.append(image_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load target image
        target_image_path = self.data[idx]
        target_image = Image.open(target_image_path).convert('L')  # Convert to grayscale

        # Convert target image to a NumPy array
        target_image = np.array(target_image)

        target_image = segmap_to_binary(target_image)

        # Generate input image by sampling pixels
        input_image = sample_pixels(target_image, self.num_samples)


        
        # Apply the transformation to the image
        input_image = self.transform(input_image)

        #   Ensure the tensor is of type float32
        input_image = input_image.to(torch.float32)

        target_image = torch.tensor(target_image, dtype=torch.float32)

        return input_image, target_image




