## Project Overview
This project aims to develop a reinforcement learning (RL) agent to navigate its path through a simulated poultry farm grid environment. The agent’s task is to find its way through the grid, avoiding obstacles, hot zones, and reaching a window while receiving rewards or penalties for its actions. The environment simulates a basic 5x5 grid with dynamic obstacles and target areas, making it ideal for experimenting with decision-making using RL. The agent uses Deep Q-Network (DQN) to learn optimal navigation strategies based on its position, obstacles, hot zones, and other grid conditions.

## Project Structure
- ``poultry_farm_env.py``: Defines the custom Gym environment for simulating poultry farm navigation.
- ``train.py``: Trains the DQN agent using the custom environment and Keras-based model.
- ``play.py``: Simulates the trained agent's behaviour in the poultry farm grid environment.
- ``requirements.txt``: Contains the necessary dependencies for the project.
- ``README.md``: Project description, setup, and usage instructions.

## Custom Environment
The PoultryFarmEnv environment simulates a 5x5 grid-based poultry farm where the agent must navigate from the top-left corner (position [0,0]) to the bottom-right corner (position [4,4]). The agent must avoid obstacles and hot zones, while aiming to reach the window position which is the target. The environment is designed to test pathfinding and decision-making strategies in a grid.

## Actions
The agent can take the following actions:
- Action 0: Move Up
- Action 1: Move Down
- Action 2: Move Left
- Action 3: Move Right

## States
Grid State: A flattened 5x5 grid representation (0 to 4 values):
- 0: Empty space
- 1: Agent’s current position
- 2: Hot Zone (dangerous area)
- 3: Window (goal to reach)
- 4: Obstacle (impassable area)

## Rewards
The agent receives rewards based on its actions:

**1. Positive Rewards:**
      - Reward +10 for reaching the window at position [4, 4] (success).
      - Reward +1 for every step taken towards the goal without colliding with obstacles or hot zones.
  
**2. Negative Rewards:**
      - Penalty -2 for stepping into a hot zone.
      - Penalty -2 for colliding with an obstacle.
      - Penalty -1 for moving out of bounds or trying invalid actions (like moving into walls).
      - Penalty -1 for excessive steps without making meaningful progress toward the goal (e.g., moving back and forth).
  
## Termination Conditions
The episode terminates when:
- The agent reaches the window at position [4, 4], in which case the episode ends successfully.
- The agent makes 15 steps (or after the set number of steps), which can result in a penalty for excessive steps.
- The agent collides with an obstacle or a hot zone that results in a penalty and causes the environment to reset.

## Running the Project
### Prerequisites :
Python 3.7+ Gym library Keras and Keras-RL

## Installation
To set up the environment and run the project, follow these steps:

1. Clone the repository:
   ```python
   git clone https://github.com/Chambeline-Nkah/poultryhealth-DQN.git
   cd poultryhealth-DQN
   ```

2. Create a virtual environment and install dependencies:
   ```python
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Run the training script to train the DQN agent:
   ```python
   python train.py
   ```
4. Simulate the trained agent's behavior:
   ```python
   python play.py
   ```

## Conclusion
This project demonstrates the application of reinforcement learning for optimizing navigation and decision-making in a grid-based poultry farm environment.

## Demo Video
[Link to video](https://www.loom.com/share/d7717caef1544a2c8b2b23b0890b73df?sid=dc7fdd9a-0345-4fb0-9a3f-b8e80326cb37)