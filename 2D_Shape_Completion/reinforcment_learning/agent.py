import gymnasium as gym
import numpy as np
from collections import defaultdict
from typing import List, Tuple


class Agent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate, and an epsilon.

        Args:
            env: The training environment
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.env = env
        # Initialize Q-values as a dictionary with a default value of zero for each action (x, y)
        self.q_values = defaultdict(lambda: np.zeros((env.action_space["action_space_border"].n, env.action_space["action_space_angle"].n)))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

        self.action = {}

    def get_action(self, obs):
        """
        Returns the best action (x, y) with probability (1 - epsilon),
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # With probability epsilon, return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # With probability (1 - epsilon) act greedily (exploit)
        else:
            q_value_matrix = self.q_values[tuple(obs)]
            best_action_indices = np.unravel_index(np.argmax(q_value_matrix), q_value_matrix.shape)
            self.action["action_space_border"], self.action["action_space_angle"] = best_action_indices
            return self.action


    def update(
        self,
        obs: np.array,
        action,
        reward: float,
        terminated: bool,
        truncated: bool,
        next_obs,
    ):
        """Updates the Q-value of an action."""
        border_action = action["action_space_border"]
        angle_action = action["action_space_angle"]

        future_q_value = (not terminated) * np.max(self.q_values[tuple(next_obs)])
        temporal_difference = (
                reward + self.discount_factor * future_q_value - self.q_values[tuple(obs)][border_action, angle_action]
                )
        self.q_values[tuple(obs)][border_action, angle_action] += self.lr * temporal_difference
        self.training_error.append(temporal_difference)


        

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)