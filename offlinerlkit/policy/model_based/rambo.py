import numpy as np
import torch
import torch.nn as nn
import gym
import os

from torch.nn import functional as F
from typing import Dict, Union, Tuple
from collections import defaultdict
from operator import itemgetter
from offlinerlkit.policy import MOPOPolicy
from offlinerlkit.dynamics import BaseDynamics


class RAMBOPolicy(MOPOPolicy):
    """
    RAMBO-RL: Robust Adversarial Model-Based Offline Reinforcement Learning <Ref: https://arxiv.org/abs/2204.12581>
    """

    def __init__(
        self,
        dynamics: BaseDynamics,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        dynamics_adv_optim: torch.optim.Optimizer, 
        tau: float = 0.005,
        gamma: float  = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2, 
        adv_weight: float=0, 
        adv_train_steps: int=1000, 
        adv_rollout_batch_size: int=256, 
        adv_rollout_length: int=5, 
        include_ent_in_adv: bool=False,   #  这里是不是False
        device="cpu"
    ) -> None:
        super().__init__(
            dynamics, 
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            alpha=alpha
        )

        self._dynmics_adv_optim = dynamics_adv_optim
        self._adv_weight = adv_weight
        self._adv_train_steps = adv_train_steps
        self._adv_rollout_batch_size = adv_rollout_batch_size
        self._adv_rollout_length = adv_rollout_length
        self._include_ent_in_adv = include_ent_in_adv
        self.device = device
        
    def load(self, path):
        self.load_state_dict(torch.load(os.path.join(path, "rambo_pretrain.pt"), map_location="cpu"))

    def pretrain(self, data: Dict, logger, n_epoch, batch_size, lr, save_path) -> None:
        self._bc_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        observations = data["observations"]
        actions = data["actions"]
        sample_num = observations.shape[0]
        idxs = np.arange(sample_num)

        logger.info("Pretraining policy")
        self.actor.train()
        for i_epoch in range(n_epoch):
            np.random.shuffle(idxs)
            sum_loss = 0
            for i_batch in range(sample_num // batch_size):
                batch_obs = observations[i_batch * batch_size: (i_batch + 1) * batch_size]
                batch_act = actions[i_batch * batch_size: (i_batch + 1) * batch_size]
                batch_obs = torch.from_numpy(batch_obs).to(self.device)
                batch_act = torch.from_numpy(batch_act).to(self.device)
                dist = self.actor(batch_obs)
                log_prob = dist.log_prob(batch_act)
                bc_loss = - log_prob.mean()

                self._bc_optim.zero_grad()
                bc_loss.backward()
                self._bc_optim.step()
                sum_loss += bc_loss.cpu().item()
            logger.log_scalar("bc_pretrain/bc_loss(neg loglikelihood)", sum_loss/i_batch, step=i_epoch)
            logger.info(f"bc_pretrain: epoch {i_epoch}, bc loss {sum_loss/i_batch}")
        if save_path is None:
            save_path = os.path.join(logger.output_path, "pretrain")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(self.state_dict(), os.path.join(save_path, "rambo_pretrain.pt"))

    def update_dynamics(
        self, 
        real_buffer, 
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        all_loss_info = {
            "adv_dynamics_update/all_loss": 0, 
            "adv_dynamics_update/sl_loss": 0, 
            "adv_dynamics_update/adv_loss": 0, 
            "adv_dynamics_update/adv_advantage": 0, 
            "adv_dynamics_update/adv_log_prob": 0, 
        }
        # self.dynamics.train()
        steps = 0
        while steps < self._adv_train_steps:
            init_obss = real_buffer.sample(self._adv_rollout_batch_size)["observations"].cpu().numpy()
            observations = init_obss
            for t in range(self._adv_rollout_length):
                actions = self.select_action(observations)
                sl_observations, sl_actions, sl_next_observations, sl_rewards = \
                    itemgetter("observations", "actions", "next_observations", "rewards")(real_buffer.sample(self._adv_rollout_batch_size))
                next_observations, terminals, loss_info = self.dynamics_step_and_forward(observations, actions, sl_observations, sl_actions, sl_next_observations, sl_rewards)
                for _key in loss_info:
                    all_loss_info[_key] += loss_info[_key]
                # nonterm_mask = (~terminals).flatten()
                steps += 1
                # observations = next_observations[nonterm_mask]
                observations = next_observations
                # if nonterm_mask.sum() == 0:
                    # break
                if steps == 1000:
                    break
        # self.dynamics.eval()
        return {_key: _value/steps for _key, _value in all_loss_info.items()}


    def dynamics_step_and_forward(
        self, 
        observations,  
        actions, 
        sl_observations, 
        sl_actions, 
        sl_next_observations, 
        sl_rewards, 
    ):
        obs_act = np.concatenate([observations, actions], axis=-1)
        obs_act = self.dynamics.scaler.transform(obs_act)
        diff_mean, logvar = self.dynamics.model(obs_act)
        observations = torch.from_numpy(observations).to(diff_mean.device)
        diff_obs, diff_reward = torch.split(diff_mean, [diff_mean.shape[-1]-1, 1], dim=-1)
        mean = torch.cat([diff_obs + observations, diff_reward], dim=-1)
        # mean[..., :-1] = mean[..., :-1] + observations
        # mean[..., :-1] += observations
        std = torch.sqrt(torch.exp(logvar))
        
        dist = torch.distributions.Normal(mean, std)
        ensemble_sample = dist.sample()
        ensemble_size, batch_size, _ = ensemble_sample.shape
        
        # select the next observations
        selected_indexes = self.dynamics.model.random_elite_idxs(batch_size)
        sample = ensemble_sample[selected_indexes, np.arange(batch_size)]
        next_observations = sample[..., :-1]
        rewards = sample[..., -1:]
        terminals = np.squeeze(self.dynamics.terminal_fn(observations.detach().cpu().numpy(), actions, next_observations.detach().cpu().numpy()))

        # compute logprob
        log_prob = dist.log_prob(sample)
        log_prob = log_prob[self.dynamics.model.elites.data, ...]
        log_prob = log_prob.exp().mean(dim=0).log().sum(-1)

        # compute the advantage
        with torch.no_grad():
            next_actions, next_policy_log_prob = self.actforward(next_observations, deterministic=True)
            next_q = torch.minimum(
                self.critic1(next_observations, next_actions), 
                self.critic2(next_observations, next_actions)
            )
            if self._include_ent_in_adv:
                next_q = next_q - self._alpha * next_policy_log_prob
            value = rewards + (1-torch.from_numpy(terminals).to(mean.device).float().unsqueeze(1)) * self._gamma * next_q

            value_baseline = torch.minimum(
                self.critic1(observations, actions), 
                self.critic2(observations, actions)
            )
            advantage = value - value_baseline
            advantage = (advantage - advantage.mean()) / (advantage.std()+1e-6)
        adv_loss = (log_prob * advantage).sum()

        # compute the supervised loss
        sl_input = torch.cat([sl_observations, sl_actions], dim=-1).cpu().numpy()
        sl_target = torch.cat([sl_next_observations-sl_observations, sl_rewards], dim=-1)
        sl_input = self.dynamics.scaler.transform(sl_input)
        sl_mean, sl_logvar = self.dynamics.model(sl_input)
        sl_inv_var = torch.exp(-sl_logvar)
        sl_mse_loss_inv = (torch.pow(sl_mean - sl_target, 2) * sl_inv_var).mean(dim=(1, 2))
        sl_var_loss = sl_logvar.mean(dim=(1, 2))
        sl_loss = sl_mse_loss_inv.sum() + sl_var_loss.sum()
        sl_loss = sl_loss + self.dynamics.model.get_decay_loss()
        sl_loss = sl_loss + 0.001 * self.dynamics.model.max_logvar.sum() - 0.001 * self.dynamics.model.min_logvar.sum()

        all_loss = self._adv_weight * adv_loss + sl_loss
        self._dynmics_adv_optim.zero_grad()
        all_loss.backward()
        self._dynmics_adv_optim.step()

        return next_observations.cpu().numpy(), terminals, {
            "adv_dynamics_update/all_loss": all_loss.cpu().item(), 
            "adv_dynamics_update/sl_loss": sl_loss.cpu().item(), 
            "adv_dynamics_update/adv_loss": adv_loss.cpu().item(), 
            "adv_dynamics_update/adv_advantage": advantage.mean().cpu().item(), 
            "adv_dynamics_update/adv_log_prob": log_prob.mean().cpu().item(), 
        }