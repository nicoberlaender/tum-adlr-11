import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from dataset import preprocessing
import numpy as np
import matplotlib.pyplot as plt

class ImageDataset(Dataset):
    def __init__(self, root_dir, num_samples = 100, len_dataset = 750, transform=transforms.ToTensor()):
        """
        Args:
            root_dir (string): Directory containing all the numbered folders with images.
            num_samples (int): Number of pixels to sample for input images.
            transform (callable, optional): Transformations to apply to the images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.num_samples = num_samples
        self.data = []
        self.sampled_data = []
        for file in os.listdir(root_dir):
            image_path = os.path.join(root_dir, file)
            if (".DS_Store" in image_path):
                continue
            if os.path.isfile(image_path):  # Check if it's a file
                for _ in range(num_samples):
                    self.data.append(image_path)
        # Gather all sampled image paths in the samples directory
        samples_dir = os.path.join(root_dir, 'samples')
        for i in range(len_dataset):
            folder_path = os.path.join(samples_dir, str(i))
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, file)
                    if os.path.isfile(image_path):
                        self.sampled_data.append(image_path)


        #get filename without folder
        self.data.sort(key=lambda x: os.path.basename(x))
        self.sampled_data.sort(key=lambda x: x)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load target image
        target_image_path = self.data[idx]
        target_image = Image.open(target_image_path).convert('L')
        input_image_path = self.sampled_data[idx]
        input_image = Image.open(input_image_path).convert('L')

        input = self.transform(input_image)
        target = self.transform(target_image)


        return input, target