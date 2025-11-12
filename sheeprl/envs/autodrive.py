from __future__ import annotations

from typing import Any, Dict, List, Optional, SupportsFloat, Tuple, Union

#import unity ml agents and gym interface
from mlagents_envs.environment import UnityEnvironment  # type: ignore
from gym_unity.envs import UnityToGymWrapper  # type: ignore
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class AutoDRIVEWrapper(gym.Wrapper):
    def __init__(self) -> None:

        # Launch Unity and wrap it
        unity_env = UnityEnvironment()
        self.env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)

        # Define observation/action spaces for the RL agent
    # Unity exports a 55-element float vector describing the car's state
    # (e.g. position, velocity, heading, lidar, etc.)
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(57,), dtype=np.float32),
            }
        )
# Two discrete action channels (each with 3 options)
# [steer, throttle] → 0=left/brake, 1=center/coast, 2=right/accelerate ?
        self.action_space = spaces.MultiDiscrete([3, 3])
# Allow rewards to be any real number (no bounds or clipping)
        self.reward_range = (-np.inf, np.inf)
# Rendering mode tells Gym to return RGB image arrays for visualization
        self._render_mode: str = "rgb_array"
# Metadata for Gym — simulator runs at 60 frames per second
        self._metadata = {"render_fps": 60}

    # Return the current render mode string
    @property
    def render_mode(self) -> str:
        return self._render_mode

   # Convert Unity’s raw observation to a Gym dictionary
    def _convert_obs(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        return {"state": obs[0]}
    # Single environment step: apply action → get next obs, reward, don
    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        return self._convert_obs(obs), reward, done, False, info
    # Reset Unity environment and return initial observation
    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        return self._convert_obs(obs), {}

    # Render a visual frame (Unity RGB array)
    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        return self.env.render()

    def close(self):
        self.env.close()
