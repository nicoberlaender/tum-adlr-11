import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from dataset import preprocessing
import numpy as np
import matplotlib.pyplot as plt

class ImageDataset(Dataset):
    def __init__(self):
        self.num_samples = 50

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # Load target image
        target_image_path = "./overfit_data/0_surface.png"
        target_image = Image.open(target_image_path).convert('L')
        target_image_array = np.array(target_image)
        input_image_binary = preprocessing.sample_pixels(target_image_array, self.num_samples)
        input_image_array = preprocessing.binary_to_image(input_image_binary)
        input_image = Image.fromarray(input_image_array) 
        #save input image
        input_image.save('./overfit_data/input_image.png')

        input = transforms.ToTensor()(input_image)
        target = transforms.ToTensor()(target_image)


        return input, target




