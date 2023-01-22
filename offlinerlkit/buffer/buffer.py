import numpy as np
import torch

from typing import Optional, Union, Tuple, Dict


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple,
        obs_dtype: np.dtype,
        action_dim: int,
        action_dtype: np.dtype,
        device: str = "cpu"
    ) -> None:
        self._max_size = buffer_size
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.action_dim = action_dim
        self.action_dtype = action_dtype

        self._ptr = 0
        self._size = 0

        self.observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.next_observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.actions = np.zeros((self._max_size, self.action_dim), dtype=action_dtype)
        self.rewards = np.zeros((self._max_size, 1), dtype=np.float32)
        self.terminals = np.zeros((self._max_size, 1), dtype=np.float32)

        self.device = torch.device(device)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        terminal: np.ndarray
    ) -> None:
        # Copy to avoid modification by reference
        self.observations[self._ptr] = np.array(obs).copy()
        self.next_observations[self._ptr] = np.array(next_obs).copy()
        self.actions[self._ptr] = np.array(action).copy()
        self.rewards[self._ptr] = np.array(reward).copy()
        self.terminals[self._ptr] = np.array(terminal).copy()

        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)
    
    def add_batch(
        self,
        obss: np.ndarray,
        next_obss: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray
    ) -> None:
        batch_size = len(obss)
        indexes = np.arange(self._ptr, self._ptr + batch_size) % self._max_size

        self.observations[indexes] = np.array(obss).copy()
        self.next_observations[indexes] = np.array(next_obss).copy()
        self.actions[indexes] = np.array(actions).copy()
        self.rewards[indexes] = np.array(rewards).copy()
        self.terminals[indexes] = np.array(terminals).copy()

        self._ptr = (self._ptr + batch_size) % self._max_size
        self._size = min(self._size + batch_size, self._max_size)
    
    def load_dataset(self, dataset: Dict[str, np.ndarray], data_limit: None) -> None:
        if data_limit is None:
            data_limit = 1e10
        observations = np.array(dataset["observations"], dtype=self.obs_dtype)
        data_limit = min(data_limit, observations.shape[0])
        
        observations = np.array(dataset["observations"], dtype=self.obs_type)[:data_limit]
        next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)[:data_limit]
        actions = np.array(dataset["actions"], dtype=self.action_dtype)[:data_limit]
        rewards = np.array(dataset["rewards"], dtype=np.float32).reshape(-1, 1)[:data_limit]
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1)[:data_limit]

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals

        self._ptr = len(observations)
        self._size = len(observations)
     
    def normalize_obs(self, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.observations.mean(0, keepdims=True)
        std = self.observations.std(0, keepdims=True) + eps
        self.observations = (self.observations - mean) / std
        self.next_observations = (self.next_observations - mean) / std
        obs_mean, obs_std = mean, std
        return obs_mean, obs_std
    
    def normalize_reward(self, eps: float = 1e-3) -> None:
        mean = self.rewards.mean(0, keepdims=True)
        std = self.rewards.std(0, keepdims=True) + eps
        self.rewards = (self.rewards - mean) / std
    
    # def normalize_reward(self) -> None:
    #     terminals_float = np.zeros_like(self.rewards)
    #     for i in range(len(terminals_float) - 1):
    #         if np.linalg.norm(self.observations[i + 1] -
    #                             self.next_observations[i]
    #                             ) > 1e-6 or self.terminals[i] == 1.0:
    #             terminals_float[i] = 1
    #         else:
    #             terminals_float[i] = 0

    #     terminals_float[-1] = 1

    #     # split_into_trajectories
    #     trajs = [[]]
    #     for i in range(len(self.observations)):
    #         trajs[-1].append((self.observations[i], self.actions[i], self.rewards[i], 1.0-self.terminals[i],
    #                         terminals_float[i], self.next_observations[i]))
    #         if terminals_float[i] == 1.0 and i + 1 < len(self.observations):
    #             trajs.append([])
        
    #     def compute_returns(traj):
    #         episode_return = 0
    #         for _, _, rew, _, _, _ in traj:
    #             episode_return += rew

    #         return episode_return

    #     trajs.sort(key=compute_returns)

    #     # normalize rewards
    #     self.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    #     self.rewards *= 1000.0

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:

        batch_indexes = np.random.randint(0, self._size, size=batch_size)
        
        return {
            "observations": torch.tensor(self.observations[batch_indexes]).to(self.device),
            "actions": torch.tensor(self.actions[batch_indexes]).to(self.device),
            "next_observations": torch.tensor(self.next_observations[batch_indexes]).to(self.device),
            "terminals": torch.tensor(self.terminals[batch_indexes]).to(self.device),
            "rewards": torch.tensor(self.rewards[batch_indexes]).to(self.device)
        }
    
    def sample_all(self) -> Dict[str, np.ndarray]:
        return {
            "observations": self.observations[:self._size].copy(),
            "actions": self.actions[:self._size].copy(),
            "next_observations": self.next_observations[:self._size].copy(),
            "terminals": self.terminals[:self._size].copy(),
            "rewards": self.rewards[:self._size].copy()
        }