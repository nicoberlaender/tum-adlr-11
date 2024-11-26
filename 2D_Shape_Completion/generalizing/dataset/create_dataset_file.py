import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from dataset import preprocessing
import numpy as np
import matplotlib.pyplot as plt

class ImageDataset(Dataset):
    def __init__(self, root_dir, num_samples = 15, len_dataset = 100, transform=transforms.ToTensor()):
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
        counter = 0
        # Gather all image paths in the root directory
        for file in os.listdir(root_dir):
            image_path = os.path.join(root_dir, file)
            if (".DS_Store" in image_path):
                continue
            if os.path.isfile(image_path) and counter<len_dataset:  # Check if it's a file
                target_image = Image.open(image_path).convert('L')
                target_image_array = np.array(target_image)
                input_image_binary = preprocessing.sample_pixels(target_image_array, self.num_samples)
                input_image_array = preprocessing.binary_to_image(input_image_binary)
                input_image = Image.fromarray(input_image_array) 
                #save input image
                sample_path = self.root_dir + '/samples/' + os.path.basename(image_path).split('.')[0] + '_sample.png'
                input_image.save(sample_path)
                self.data.append(image_path)
                self.sampled_data.append(sample_path)
                counter = counter+1


        #get filename without folder
        self.data.sort(key=lambda x: os.path.basename(x))
        self.sampled_data.sort(key=lambda x: os.path.basename(x))


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




