import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ContinuousCritic(nn.Module):
    """Continuous soft Q-network model for continous SAC."""

    def __init__(self, env):
        """Initialize the critic model.

        Parameters
        ----------
        env : gym environment
            Gym environment being used for learning.
        """
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.observation_space.shape).prod()
            + np.prod(env.action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, states, actions):
        """
        Calculates the Q-values of state-action pairs.

        Parameters
        ----------
        states : tensor
            States or observations.
        actions : tensor
            Actions.

        Returns
        -------
        q_values : tensor
            Q-values for given state-action pairs.
        """
        x = torch.cat([states, actions], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)

        return q_values


class DiscreteCritic(nn.Module):
    """Discrete soft Q-network model for discrete SAC with discrete actions
    and continuous observations."""

    def __init__(self, model_config):
        """Initialize the critic model.

        Parameters
        ----------
        env : gym environment
            Gym environment being used for learning.
        """
        super().__init__()
        self.fc1 = nn.Linear(12, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, model_config["output_size"])

    def forward(self, states):
        """
        Calculates Q-values for each state-action.

        Parameters
        ----------
        states : tensor
            States or observations.

        Returns
        -------
        q_values : tensor
            Q-values for all actions possible with input state.
        """
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)

        return q_values


class DiscreteCriticDiscreteObs(nn.Module):
    """Discrete soft Q-network model with discrete actions and discrete
    observations.
    """

    def __init__(self, model_config):
        """Initialize model.

        Args:
            model_config: Dictionary containing model configuration
                parameters.
        """
        super().__init__()
        self.embedding = nn.Embedding(32, 8)
        self.fc2 = nn.Linear(96, 64)
        self.fc3 = nn.Linear(64, model_config["output_size"])

    def forward(self, states):
        """Calculate Q-values.

        Args:
            states: States or observations.

        Returns:
            Q-values for actions.
        """
        x = self.embedding(states)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)

        return q_values
