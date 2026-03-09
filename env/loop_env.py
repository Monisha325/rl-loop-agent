
import gymnasium as gym
import numpy as np
import json
from rewards.reward_function import compute_reward

class LoopDistributionEnv(gym.Env):

    def __init__(self, sdg_json):

        with open(sdg_json) as f:
            data = json.load(f)

        self.nodes = data["nodes"]
        self.node_labels = [n["id"] for n in self.nodes]

        self.index = 0

        self.observation_space = gym.spaces.Box(
            low=0,
            high=100,
            shape=(4,),
            dtype=np.float32
        )

        self.action_space = gym.spaces.Discrete(2)

    def reset(self, seed=None, options=None):

        self.index = 0
        return self._get_state(), {}

    def _get_state(self):

        node = self.nodes[self.index]

        return np.array([
            node["size"],
            int(node["parallelizable"]),
            node["loop_carried_count"],
            self.index / len(self.nodes)
        ], dtype=np.float32)

    def step(self, action):

        node = self.nodes[self.index]

        reward = compute_reward(node, action)

        self.index += 1

        done = self.index >= len(self.nodes)

        if done:
            next_state = np.zeros(4)
        else:
            next_state = self._get_state()

        return next_state, reward, done, False, {}
