#!/usr/bin/env python3
"""playing the game"""


import pygame
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from stable_baselines3 import DQN
from poultry_farm_env import PoultryFarmEnv

# initializing PyGame
pygame.init()

GRID_SIZE = 5
CELL_SIZE = 100
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
GREY = (200, 200, 200)

# loading the trained model
model = DQN.load("policy")

# initializing the poultry farm environment
env = PoultryFarmEnv()

# initializing the PyGame screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Poultry Farm Simulation")

# drawing the grid
def draw_grid(state, agent_pos, hot_zones, window_positions, obstacles):
    """drawing the grid"""
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            cell_rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        
            pygame.draw.rect(screen, BLACK, cell_rect, 1)

    # drawing the grey grids which represent the obstacles
    for pos in obstacles:
        pygame.draw.rect(screen, GREY, (pos[1] * CELL_SIZE, pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # drawing the green grid which represents the window (the target)
    for pos in window_positions:
        pygame.draw.rect(screen, GREEN, (pos[1] * CELL_SIZE, pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # drawing the red grids which represent the hot zones of high temperature
    for pos in hot_zones:
        pygame.draw.rect(screen, RED, (pos[1] * CELL_SIZE, pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # drawing the agent which is represented by a blue circle
    pygame.draw.circle(screen, BLUE, 
                       (agent_pos[1] * CELL_SIZE + CELL_SIZE // 2, agent_pos[0] * CELL_SIZE + CELL_SIZE // 2), 
                       CELL_SIZE // 4)

# simulation loop
def run_simulation():
    """running the simulation"""
    done = False
    state, _ = env.reset()
    clock = pygame.time.Clock()

    while not done:
        screen.fill(WHITE)

        # flattening the state for the model
        flattened_state = state.flatten()
        print(f"State: {flattened_state}")

        # using the saved model to predict the next action
        action, _states = model.predict(flattened_state, deterministic=True)
        print(f"Predicted action: {action}")
        
        state, reward, terminated, truncated, _ = env.step(action)


        print(f"Agent position: {env.agent_pos}")

        # getting the current positions
        agent_pos = env.agent_pos
        hot_zones = env.hot_zones
        window_positions = env.window_positions
        obstacles = env.obstacles

        # drawing the grid
        draw_grid(state, agent_pos, hot_zones, window_positions, obstacles)

        # displaying the agent's reward at the buttom of the grid
        font = pygame.font.Font(None, 36)
        reward_text = font.render(f"Reward: {reward}", True, BLACK)
        screen.blit(reward_text, (10, HEIGHT - 30))

        # updating the display
        pygame.display.flip()

        # slowing down the simulation to see the agent's movement
        clock.tick(2)

        # exiting the simulation if PyGame is closed
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        done = terminated or truncated

# runing the simulation
if __name__ == "__main__":
    run_simulation()
