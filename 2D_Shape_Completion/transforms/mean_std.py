from torch.utils.data import DataLoader


def calculate_mean_std(dataset, batch_size=32):
    """
    Calculate the mean and standard deviation of the dataset.

    Args:
        dataset (Dataset): The dataset instance.
        num_samples (int): Number of samples to estimate the statistics.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        tuple: Mean and standard deviation of the dataset.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    mean_sum = 0
    std_sum = 0
    total_pixels = 0

    for inputs, _ in loader:
        # Convert to tensor if not already
        inputs = inputs.view(inputs.size(0), -1)  # Flatten images (batch_size, height * width)
        mean_sum += inputs.float().mean(1).sum().item()
        std_sum += inputs.float().std(1).sum().item()
        total_pixels += inputs.size(0)

    mean = mean_sum / total_pixels
    std = std_sum / total_pixels
    return 0,1