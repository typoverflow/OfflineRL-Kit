import copy
from UtilsRL.env.wrapper.mujoco_wrapper import MujocoParamOverWrite
import torch
import numpy as np

def eval_actor(env, actor, device, n_episodes, seed, score_func=None):
    if score_func is None:
        score_func = env.get_normalized_score
    env.seed(seed)
    actor.eval()
    actor.to(device)
    episode_lengths = []
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        episode_length = 0.0
        while not done:
            action = actor.select_action(state, deterministic=True)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
        episode_rewards.append(score_func(episode_reward)*100)
        episode_lengths.append(episode_length)
    
    actor.train()
    episode_rewards = np.asarray(episode_rewards)
    episode_lengths = np.asarray(episode_lengths)
    return {
        "normalized_score_mean": episode_rewards.mean(), 
        "normalized_score_std": episode_rewards.std(), 
        "length_mean": episode_lengths.mean(), 
        "length_std": episode_lengths.std()
    }
        
    

def test_actors(actor, ckpts, raw_env, perturb_type, perturb_amp, device):
    perturb_env = copy.deepcopy(raw_env)
    perturb_env = MujocoParamOverWrite(perturb_env, {perturb_type: perturb_amp}, do_scale=True)
    eval_dict = {}
    for ckpt in ckpts:
        actor.load_state_dict(torch.load(ckpt, map_location="cpu"))
        this_eval_dict = eval_actor(perturb_env, actor, device=device, n_episodes=10, seed=0, score_func=raw_env.get_normalized_score)
        for _key, _value in this_eval_dict.items():
            eval_dict[_key] = eval_dict.get(_key, 0) + this_eval_dict[_key]
    return {_key: _value/len(ckpts) for _key, _value in eval_dict.items()}

