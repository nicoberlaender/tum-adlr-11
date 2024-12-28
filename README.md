# Shape Reconstruction with CNN and Reinforcement Learning

This project focuses on shape reconstruction using a combination of a Convolutional Neural Network (CNN) and Reinforcement Learning (RL). The project has two main components:

1. **CNN (U-Net)**: For reconstructing shapes from partial data.
2. **Reinforcement Learning (PPO)**: For determining the most informative point to explore, aiding the reconstruction process.

---

## Project Structure

### Directories and Files
- **`data/`**: Contains the dataset used for training and validation.
- **`dataset/`**: Implements the custom dataset class for loading and preprocessing data.
- **`main.py`**: The main script for training the CNN model.
- **`saved_models/`**: Pre-trained models for the shape reconstruction task.
- **`ppo/`**: Scripts and modules related to the PPO implementation for selecting exploration points.

### Key Components
#### 1. **U-Net for Shape Reconstruction**
   - The U-Net is designed to take partial shape data as input and reconstruct the complete shape.
   - Training data includes geometric shapes, such as ellipses and trapezoids, with partial visibility to simulate real-world conditions.
   - The input consists of:
     - Partial views of the shape.
     - Sparse tactile information (if available).

#### 2. **PPO for Informative Exploration**
   - Uses Proximal Policy Optimization (PPO), a popular RL algorithm, to identify the next optimal exploration point.
   - The agent iteratively improves reconstruction accuracy by focusing on unexplored or ambiguous regions of the shape.

---

## Installation

To run the project, ensure you have the following dependencies installed:

- Python >= 3.9
- PyTorch
- NumPy
- Matplotlib
- Gym
- SWIG
- Stable-Baselines3
- OS (built-in Python module)

### Setting up the Environment
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>

