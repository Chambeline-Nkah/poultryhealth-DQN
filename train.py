#!/usr/bin/env python3
"""training the agent"""


# from stable_baselines3 import DQN
# from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import gymnasium as gym
from poultry_farm_env import PoultryFarmEnv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

# initializing the poultry farm environment
env = PoultryFarmEnv()
nb_actions = env.action_space.n

# training the agent
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(24, activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(nb_actions, activation="linear"))
print(model.summary())

# compiling
policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model, 
               nb_actions=nb_actions, 
               memory=memory, 
               nb_steps_warmup=10, 
               target_model_update=1e-2, 
               policy=policy)
dqn.compile(Adam(learning_rate=1e-3), metrics=["mae"])

# training the agent
dqn.fit(env, nb_steps=100000, visualize=False, verbose=1)

# saving the model
dqn.save_weights("dqn_weights.h5f", overwrite=True)

# evaluating the trained model
scores = dqn.test(env, nb_episodes=10, visualize=False)
print(f"Mean reward: {np.mean(scores.history['episode_reward'])}, Std: {np.std(scores.history['episode_reward'])}")
