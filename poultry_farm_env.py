#!/usr/bin/env python3
"""poultry farm custom environment"""


import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time

class PoultryFarmEnv(gym.Env):
    """custom environment for managing the poultry farm"""
    def __init__(self):
        super(PoultryFarmEnv, self).__init__()
        
        # 5x5 grid
        self.grid_size = 5
        # position of agent that is [0,0]
        self.agent_pos = [0, 0]
        # position of the window to be opened
        self.window_positions = [(1, 1)]
        # hot zone (high temperatures) areas in the poultry farm
        self.hot_zones = [(2, 2), (3, 3)]
        # obstacles such as walls in the poultry farm
        self.obstacles = [(1, 1), (1, 3), (3, 1)]
        # maximum number of steps to be taken by the agent in an episode to reach the target
        self.max_steps = 15
        self.current_steps = 0

        # defining the action spaces of the agent
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        
        # flattened grid representation
        # 0: Empty, 1: Agent, 2: Hot Zone, 3: Window, 4: Obstacle
        self.observation_space = spaces.Box(
            low=0,
            high=4,
            shape=(self.grid_size * self.grid_size,),
            dtype=np.int32
        )

    def reset(self, seed=None, options=None):
        """resetting the environment to its initial state"""
        super().reset(seed=seed)
        np.random.seed(seed)

        # resetting the environment
        self.state = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.agent_pos = [0, 0]  # resetting the agent position
        self.state[tuple(self.agent_pos)] = 1

        self.current_steps = 0  # resetting step counter

        return self.state.flatten(), {}

    def step(self, action):
        """performing an action in the poultry farm"""
        reward = 0
        terminated = False  
        truncated = False

        # moving up
        if action == 0:  
            if self.agent_pos[0] > 0:
                self.agent_pos[0] -= 1
            else:
                reward -= 1  # penalty for trying to move out of bounds

        # moving down
        elif action == 1:
            if self.agent_pos[0] < self.grid_size - 1:
                self.agent_pos[0] += 1
            else:
                reward -= 1

        # moving left
        elif action == 2:
            if self.agent_pos[1] > 0:
                self.agent_pos[1] -= 1
            else:
                reward -= 1

        # moving right
        elif action == 3:  
            if self.agent_pos[1] < self.grid_size - 1:
                self.agent_pos[1] += 1
            else:
                reward -= 1

        # checking for collision with hot zones or obstacles
        if tuple(self.agent_pos) in self.hot_zones:
            reward -= 2  # penalty for stepping in a grid containing a hot zone
        elif tuple(self.agent_pos) in self.obstacles:
            reward -= 2  # penalty for stepping in a grid containing an obstacle

        # checking if the agent has reached the target that is the window (green grid)
        if tuple(self.agent_pos) == self.window_positions[0]:
            reward += 10  # Reward for reaching the window
            terminated = True  # once reached, the episode is ended

        # updating the grid state
        self.state = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.state[tuple(self.agent_pos)] = 1  # updating the agent position
        for pos in self.hot_zones:
            self.state[pos] = 2
        for pos in self.window_positions:
            self.state[pos] = 3
        for pos in self.obstacles:
            self.state[pos] = 4 

        # increasing step counter and check termination conditions
        self.current_steps += 1
        if self.current_steps >= self.max_steps:
            truncated = True

        # adding a delay of 2s to slow down the agent's movement
        time.sleep(2)

        return self.state.flatten(), reward, terminated, truncated, {}

    def render(self, mode="human"):
        """using a basic grid"""
        print("\n".join(" ".join(str(cell) for cell in row) for row in self.state))
