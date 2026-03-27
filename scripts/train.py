"""PPO training for OGE (red_sat control)."""

import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

import torch
from copy import deepcopy
from skrl.agents.torch.ppo import PPO
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from oge_py import OGEEnv
from modules.env_wrapper import OGESingleEnvWrapper
from modules.networks import Policy, Value
from configs.env_cfg import env_cfg
from configs.ppo_cfg import ppo_cfg as base_ppo_cfg, trainer_cfg


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Environment
    raw_env = OGEEnv(env_cfg)
    env = OGESingleEnvWrapper(raw_env, env_cfg)

    # Deep copy config to avoid modifying the imported dict
    ppo_cfg = deepcopy(base_ppo_cfg)

    # Models
    policy = Policy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        dv_max=env_cfg.dv_max_per_step_red,
    )
    value = Value(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
    models = {"policy": policy, "value": value}

    # Memory
    memory = RandomMemory(
        memory_size=ppo_cfg["rollouts"],
        num_envs=env.num_envs,
        device=device,
    )

    # Preprocessor kwargs
    ppo_cfg["state_preprocessor_kwargs"] = {"size": env.observation_space.shape[0], "device": device}
    ppo_cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    # Agent
    agent = PPO(
        models=models,
        memory=memory,
        cfg=ppo_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    # Trainer
    trainer = SequentialTrainer(env=env, agents=agent, cfg=trainer_cfg)

    print("Starting PPO training on OGE (red_sat)")
    trainer.train()


if __name__ == "__main__":
    main()