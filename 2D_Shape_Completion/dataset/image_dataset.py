import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=transforms.ToTensor(), test=False):
        """
        Args:
            root_dir (string): Directory containing all the numbered folders with images.
            num_samples (int): Number of pixels to sample for input images.
            len_dataset (int): Maximum number of samples to take from the dataset.
            transform (callable, optional): Transformations to apply to the images.
        """
        if test:
            root_dir = os.path.join(root_dir, 'test')
        else:
            root_dir = os.path.join(root_dir, 'train')
        samples_dir = os.path.join(root_dir, 'samples')
        self.transform = transform
        self.data = []
        self.sampled_data = []
        
        for file in os.listdir(root_dir):
            image_path = os.path.join(root_dir, file)
            if ".DS_Store" in image_path:
                continue
            if os.path.isfile(image_path):  # Check if it's a file
                number = os.path.basename(image_path).split('.')[0]
                folder_path = os.path.join(samples_dir, str(number))
                num_samples = len(os.listdir(folder_path))

                for _ in range(num_samples):
                    self.data.append(image_path)

                if os.path.isdir(folder_path):
                    for file in os.listdir(folder_path):
                        image_path = os.path.join(folder_path, file)
                        if os.path.isfile(image_path):
                            self.sampled_data.append(image_path)
            assert(len(self.data) == len(self.sampled_data))

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