"""FetchPush simulator adapter: flat observations, 3-D gripper actions and the optional approach reward."""
import json
from pathlib import Path
import numpy as np
import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)
ENV = 'FetchPushDense-v4'

class RobotObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (28,), np.float32)

    def observation(self, obs):
        result = np.concatenate((obs['observation'], obs['desired_goal'])).astype(np.float32)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError('Nonfinite robot observation')
        return result

class RobotAction(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(-1, 1, (3,), np.float32)

    def action(self, action):
        return np.append(np.asarray(action, np.float32), np.float32(0))


class RobotReward(gym.Wrapper):
    def __init__(self, env, approach_weight=0.):
        super().__init__(env)
        self.approach_weight = float(approach_weight)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.initial_block = obs[3:6].copy()
        return obs, info

    def step(self, action):
        obs, native, terminated, truncated, info = self.env.step(action)
        distance = float(np.linalg.norm(obs[:3]-obs[3:6]))
        reward = float(native)-self.approach_weight*distance
        info = dict(info, native_reward=float(native), shaped_reward=reward,
            approach_distance_cm=100*distance,
            block_displacement_cm=float(100*np.linalg.norm(obs[3:6]-self.initial_block)))
        return obs, reward, terminated, truncated, info


def make_env(approach_weight=0., **kwargs):
    env = RobotAction(RobotObservation(gym.make(ENV, **kwargs)))
    return RobotReward(env, approach_weight)


def write_json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)
