# Shape Reconstruction with CNN and Reinforcement Learning

This project focuses on shape reconstruction using a combination of a Convolutional Neural Network (CNN) and Reinforcement Learning (RL). The project has two main components:

1. **CNN (U-Net)**: For reconstructing shapes from sparse data.
2. **Reinforcement Learning (PPO)**: For determining the most informative point to explore, aiding the reconstruction process.

---

## Project Structure

### Directories and Files
- **`2D_Shape_Completion/`**: Contains code for training and evaluating the shape completion model.
- **`Reinforcement_Learning`**: Contains code to define the environment as well as train and evaluate the RL agent.

The data for the shape completion task can be found [here](https://drive.google.com/file/d/1V-r0bhskPLhFb2RKnoXtLhla_X0ossbe/view?usp=share_link) and should be extracted to `2D_Shape_Completion/data`.
The data for the RL task can be found [here](https://drive.google.com/file/d/19Ffqj8n5J1-biwyEPVk5ulh2ruUav1_i/view?usp=share_link) and should be extracted to `Reinforcement_Learning/data`.

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

### Setting up the Environment
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
2. Create a virtual environment using Conda or venv.
3. ```pip install -r requirements.txt```

