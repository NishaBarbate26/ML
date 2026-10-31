# AIM: Solve the Taxi Problem Using Reinforcement Learning.

# Installation commands:
# Windows:
# python -m pip install gymnasium numpy
# python practical_taxi.py
#
# Ubuntu/Linux:
# pip install gymnasium numpy
# python3 -m pip install gymnasium numpy
# python3 practical_taxi.py

import gymnasium as gym
import numpy as np

# Create Taxi environment

env = gym.make("Taxi-v4")

# Get state and action information

state_size = env.observation_space.n
action_size = env.action_space.n

print("State Space:", state_size)
print("Action Space:", action_size)

# Create Q-table

q_table = np.zeros((state_size, action_size))

# Learning parameters

learning_rate = 0.7
discount_factor = 0.9
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01

episodes = 10000

# Training the agent

for episode in range(episodes):

    state, info = env.reset()

    done = False

    while not done:

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        q_table[state, action] = q_table[state, action] + learning_rate * (
            reward + discount_factor * np.max(q_table[next_state])
            * (not done) - q_table[state, action]
        )

        state = next_state

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

# Testing the trained agent

print("\nTesting Trained Taxi Agent:")

state, info = env.reset()

done = False
total_reward = 0
steps = 0

while not done:

    action = np.argmax(q_table[state])

    state, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

    total_reward += reward
    steps += 1

    print("Step:", steps, "Reward:", reward)

    if steps >= 200:
        break

print("\nTotal Reward:", total_reward)

print("Total Steps:", steps)

print("\nLearned Q-Table:")
print(q_table)

# Display Taxi environment

env.close()


# ============================================================
# EXPLANATION LINE BY LINE
# ============================================================

# import gymnasium as gym
# Imports Gymnasium to create the Taxi environment.

# import numpy as np
# Imports NumPy for numerical calculations.

# gym.make("Taxi-v3")
# Creates the Taxi environment.

# env.observation_space.n
# Returns the total number of possible states.

# env.action_space.n
# Returns the total number of possible actions.

# np.zeros()
# Creates the Q-table with zero initial values.

# learning_rate
# Controls how quickly the agent learns.

# discount_factor
# Determines the importance of future rewards.

# epsilon
# Controls the exploration rate.

# epsilon_decay
# Gradually decreases random exploration.

# min_epsilon
# Defines the minimum exploration rate.

# episodes
# Specifies the number of training episodes.

# env.reset()
# Resets the environment and returns the initial state.

# env.action_space.sample()
# Selects a random action.

# np.random.random()
# Helps decide whether to explore or use learned knowledge.

# np.argmax(q_table[state])
# Selects the action with the highest Q-value.

# env.step(action)
# Executes the selected action in the environment.

# next_state
# Stores the state reached after taking an action.

# reward
# Stores the reward received from the environment.

# terminated
# Indicates whether the task has reached a terminal state.

# truncated
# Indicates whether the episode ended because of a time limit.

# q_table update
# Updates the Q-value using the Q-Learning formula.

# epsilon * epsilon_decay
# Reduces exploration after each episode.

# max()
# Ensures that epsilon does not fall below its minimum value.

# Testing the trained agent
# Runs the trained agent without random exploration.

# total_reward
# Stores the total reward received during testing.

# steps
# Counts the number of actions taken.

# env.close()
# Closes the Taxi environment.