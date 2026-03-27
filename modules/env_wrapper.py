"""Single-agent wrapper for OGE + skrl PPO."""

from __future__ import annotations
from typing import Any
import numpy as np
import torch
import gymnasium
from skrl.envs.wrappers.torch.base import Wrapper


class OGESingleEnvWrapper(Wrapper):
    """Wraps OGEEnv for single-agent PPO (controls red_sat)."""

    def __init__(self, env, cfg) -> None:
        super().__init__(env)
        self._oge = env.oge
        self._dv_max_blue = cfg.dv_max_per_step_blue

        # Blue sat discrete action set
        dv = self._dv_max_blue
        self._blue_actions = np.array([
            [dv, 0, 0], [-dv, 0, 0], [0, dv, 0],
            [0, -dv, 0], [0, 0, dv], [0, 0, -dv], [0, 0, 0]
        ], dtype=np.float32)

        self._obs_size = self._oge.get_obs_size()
        self._single_obs_size = self._obs_size  # Each agent has obs_size dimensions
        self._last_obs = None

        # Override spaces for single red_sat agent
        self._observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._single_obs_size,), dtype=np.float32
        )
        self._action_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def num_envs(self) -> int:
        return 1

    def reset(self, seed=None, options=None):
        obs_np, info = self._env.reset(seed=seed, options=options)
        self._last_obs = np.asarray(obs_np, dtype=np.float32)
        # Extract red_sat obs (second row)
        red_obs = self._last_obs[1]
        return torch.tensor(red_obs, dtype=torch.float32).unsqueeze(0).to(self.device), info

    def step(self, actions):
        red_action = actions.squeeze().cpu().numpy()
        # Blue sat uses random action
        blue_action = self._blue_actions[np.random.randint(len(self._blue_actions))]
        combined_actions = np.vstack([blue_action, red_action])
        self._oge.act(combined_actions)
        obs_np = self._oge.get_observations()
        # 如果要自定义奖励函数，可以修改这里，不使用self._oge.get_rewards
        rew_np = np.asarray(self._oge.get_rewards(combined_actions), dtype=np.float32)
        terminated = bool(self._oge.is_terminal())
        truncated = bool(self._oge.is_truncated())
        info = {"current_time": self._oge.get_current_time()}

        self._last_obs = np.asarray(obs_np, dtype=np.float32)
        # Extract red_sat obs (second row)
        red_obs = self._last_obs[1]
        obs_t = torch.tensor(red_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        reward_red = float(rew_np[1])  # get red_sat's reward
        rew_t = torch.tensor([[reward_red]], dtype=torch.float32).to(self.device)
        term_t = torch.tensor([[terminated]], dtype=torch.bool).to(self.device)
        trunc_t = torch.tensor([[truncated]], dtype=torch.bool).to(self.device)

        return obs_t, rew_t, term_t, trunc_t, info

    def render(self, *args, **kwargs):
        pass

    def close(self):
        pass
