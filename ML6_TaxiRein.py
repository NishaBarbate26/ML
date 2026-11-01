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
# AIM:
# Implement Q-Learning Algorithm Using the Taxi Environment
# in Gymnasium.
# ============================================================


# ============================================================
# 1. INSTALLATION AND EXECUTION COMMANDS
# ============================================================

# Windows:
# python -m pip install numpy gymnasium
# python practical_taxi.py

# Ubuntu/Linux:
# pip install numpy gymnasium
# python3 -m pip install numpy gymnasium
# python3 practical_taxi.py


# ============================================================
# 2. IMPORTING LIBRARIES
# ============================================================

# import gymnasium as gym
# Imports the Gymnasium library.
# Gymnasium provides ready-made environments for reinforcement learning.
# It is used to create the Taxi environment.

# import numpy as np
# Imports NumPy for numerical calculations and array operations.
# np is the short name used for NumPy.
# It is used to create and update the Q-Table.


# ============================================================
# 3. CREATING THE TAXI ENVIRONMENT
# ============================================================

# env = gym.make("Taxi-v4")
# Creates the Taxi environment using Gymnasium.
#
# The Taxi environment is a reinforcement learning problem.
# A taxi must pick up a passenger and drop the passenger
# at the correct destination.
#
# The environment provides:
# States: Different situations of the taxi.
# Actions: Possible movements and passenger operations.
# Rewards: Feedback for the actions performed.
#
# Important version note:
# In current Gymnasium installations, the standard Taxi
# environment ID is generally "Taxi-v3".
# "Taxi-v4" may not be available and can cause an error.
# If you get an Environment ID not found error,
# replace "Taxi-v4" with "Taxi-v3".


# ============================================================
# 4. GETTING STATE AND ACTION INFORMATION
# ============================================================

# state_size = env.observation_space.n
# Gets the total number of possible states in the environment.
# observation_space represents the possible observations.
# .n gives the total number of discrete states.
#
# In the standard Taxi environment:
# There are 500 possible states.


# action_size = env.action_space.n
# Gets the total number of possible actions.
# action_space represents the available actions.
# .n gives the total number of discrete actions.
#
# In the Taxi environment:
# There are 6 possible actions.


# print("State Space:", state_size)
# Displays the total number of possible states.


# print("Action Space:", action_size)
# Displays the total number of possible actions.


# ============================================================
# 5. CREATING THE Q-TABLE
# ============================================================

# q_table = np.zeros((state_size, action_size))
# Creates a Q-Table filled with zeros.
#
# Rows represent different states.
# Columns represent possible actions.
#
# For the standard Taxi environment:
# 500 states x 6 actions = 3000 Q-values.
#
# Initially, all Q-values are zero.
# During training, the agent updates these values
# based on rewards and future expected rewards.
#
# The Q-Table helps the agent select the best action
# for a particular state.


# ============================================================
# 6. DEFINING LEARNING PARAMETERS
# ============================================================

# learning_rate = 0.7
# Defines the learning rate, represented by alpha.
#
# It controls how much new information affects
# the existing Q-value.
#
# A higher learning rate gives more importance
# to newly received information.


# discount_factor = 0.9
# Defines the discount factor, represented by gamma.
#
# It determines the importance of future rewards.
#
# A value close to 1 gives more importance to future rewards.
# A value close to 0 gives more importance to immediate rewards.


# epsilon = 1.0
# Defines the initial exploration probability.
#
# Epsilon controls how often the agent chooses random actions.
#
# epsilon=1.0 means the agent initially explores completely.
# It selects random actions during the beginning of training.


# epsilon_decay = 0.995
# Defines the rate at which epsilon decreases.
#
# After every episode, epsilon is multiplied by 0.995.
# This gradually reduces random exploration.
#
# As training progresses, the agent uses its learned knowledge
# more frequently.


# min_epsilon = 0.01
# Defines the minimum exploration probability.
#
# Epsilon will not decrease below 0.01.
# This maintains a small amount of exploration.


# episodes = 10000
# Defines the total number of training episodes.
#
# One episode is one complete attempt by the taxi agent.
# An episode ends when the task is terminated
# or the environment truncates the episode.
#
# The agent gets 10,000 training episodes.


# ============================================================
# 7. TRAINING THE TAXI AGENT
# ============================================================

# for episode in range(episodes):
# Starts the training loop.
# The loop runs 10,000 times.
# Each repetition represents one training episode.


# ============================================================
# 8. RESETTING THE ENVIRONMENT
# ============================================================

# state, info = env.reset()
# Resets the Taxi environment at the beginning of an episode.
#
# state:
# Represents the initial state of the taxi environment.
#
# info:
# Contains additional information provided by Gymnasium.
#
# reset() returns both state and info in Gymnasium.


# ============================================================
# 9. INITIALIZING THE EPISODE STATUS
# ============================================================

# done = False
# Indicates that the current episode is not finished.
#
# It becomes True when the episode is terminated
# or truncated.


# ============================================================
# 10. STARTING THE ACTION LOOP
# ============================================================

# while not done:
# Repeats the agent's actions until the episode finishes.
#
# The taxi continues selecting actions
# until the task is terminated or truncated.


# ============================================================
# 11. SELECTING ACTION USING EPSILON-GREEDY STRATEGY
# ============================================================

# if np.random.random() < epsilon:
# Generates a random number between 0 and 1.
# Checks whether it is less than epsilon.
#
# If the condition is true, the agent explores
# by selecting a random action.


#     action = env.action_space.sample()
# Selects a random action from the available actions.
# sample() randomly selects one valid action.
#
# This is called exploration.
# The agent tries different movements and operations.


# else:
# Executes when the random number is not less than epsilon.
# The agent uses the knowledge stored in the Q-Table.


#     action = np.argmax(q_table[state])
# Selects the action with the highest Q-value
# for the current state.
#
# np.argmax() returns the index of the maximum value.
# This is called exploitation.
#
# Exploration means trying random actions.
# Exploitation means choosing the best-known action.


# ============================================================
# 12. PERFORMING THE SELECTED ACTION
# ============================================================

# next_state, reward, terminated, truncated, info = env.step(action)
# Performs the selected action in the Taxi environment.
#
# env.step(action) executes the action and returns five values.
#
# next_state:
# The new state after performing the action.
#
# reward:
# The reward or penalty received from the environment.
#
# terminated:
# True if the task has reached a terminal state.
# For example, the passenger has been successfully dropped
# at the correct destination.
#
# truncated:
# True if the episode has been stopped because of a time limit
# or another external stopping condition.
#
# info:
# Additional information provided by the environment.


# ============================================================
# 13. CHECKING WHETHER THE EPISODE IS FINISHED
# ============================================================

# done = terminated or truncated
# Combines the two episode-ending conditions.
#
# If either terminated or truncated is True,
# done becomes True.
#
# The training episode then ends.


# ============================================================
# 14. UPDATING THE Q-TABLE
# ============================================================

# q_table[state, action] = q_table[state, action] + learning_rate * (
#     reward + discount_factor * np.max(q_table[next_state])
#     * (not done) - q_table[state, action]
# )

# Updates the Q-value for the current state and selected action.
#
# Q-Learning update formula:
#
# Q(s,a) = Q(s,a) + alpha [
#     reward + gamma * max Q(next_state, next_action)
#     - Q(s,a)
# ]
#
# In this program, (not done) is also used:
#
# If done is False:
# not done becomes True, which is treated as 1.
# The future Q-value is included in the update.
#
# If done is True:
# not done becomes False, which is treated as 0.
# The future Q-value is not included.
#
# This prevents the agent from considering future rewards
# after the episode has ended.
#
# Meaning of each term:
#
# q_table[state, action]:
# Current Q-value for the selected state and action.
#
# learning_rate:
# Controls how much the Q-value is updated.
# Its value is 0.7.
#
# reward:
# Immediate reward received after the action.
#
# discount_factor:
# Controls the importance of future rewards.
# Its value is 0.9.
#
# np.max(q_table[next_state]):
# Finds the highest Q-value in the next state.
#
# The agent uses this formula to improve its decisions
# based on experience.


# ============================================================
# 15. MOVING TO THE NEXT STATE
# ============================================================

# state = next_state
# Updates the current state.
# The next action will be selected using this new state.


# ============================================================
# 16. REDUCING EXPLORATION PROBABILITY
# ============================================================

# epsilon = max(min_epsilon, epsilon * epsilon_decay)
# Reduces epsilon after every training episode.
#
# epsilon * epsilon_decay:
# Multiplies the current exploration probability by 0.995.
#
# min_epsilon:
# Represents the minimum allowed exploration probability.
#
# max() selects the larger value.
# Therefore, epsilon never decreases below 0.01.
#
# During early training:
# The agent explores more frequently.
#
# During later training:
# The agent exploits its learned knowledge more frequently.


# ============================================================
# 17. TESTING THE TRAINED TAXI AGENT
# ============================================================

# print("\nTesting Trained Taxi Agent:")
# Displays the heading for the testing phase.
#
# Testing checks how well the agent performs
# after completing the training process.


# ============================================================
# 18. RESETTING THE ENVIRONMENT FOR TESTING
# ============================================================

# state, info = env.reset()
# Resets the Taxi environment for a new testing attempt.
# The agent starts from a new initial state.


# done = False
# Indicates that the testing episode is not finished.


# total_reward = 0
# Initializes the total reward to zero.
# Rewards received during testing are added to this variable.


# steps = 0
# Initializes the step counter.
# It counts the number of actions performed during testing.


# ============================================================
# 19. PERFORMING TESTING ACTIONS
# ============================================================

# while not done:
# Repeats actions until the testing episode finishes.


# ============================================================
# 20. SELECTING THE BEST ACTION
# ============================================================

# action = np.argmax(q_table[state])
# Selects the action with the highest Q-value
# for the current state.
#
# During testing, the agent does not use random exploration.
# It uses the knowledge learned during training.


# ============================================================
# 21. EXECUTING THE ACTION
# ============================================================

# state, reward, terminated, truncated, info = env.step(action)
# Executes the selected action.
#
# The returned state becomes the new current state.
# The environment also returns a reward and episode status.


# ============================================================
# 22. CHECKING THE TESTING STATUS
# ============================================================

# done = terminated or truncated
# Checks whether the testing episode has finished.
# It becomes True if the task is terminated or truncated.


# ============================================================
# 23. CALCULATING TOTAL REWARD
# ============================================================

# total_reward += reward
# Adds the current reward to the total reward.
#
# This helps calculate the total performance
# of the taxi agent during testing.


# ============================================================
# 24. COUNTING THE NUMBER OF STEPS
# ============================================================

# steps += 1
# Increases the step counter by one.
# Each action performed by the taxi counts as one step.


# ============================================================
# 25. DISPLAYING STEP INFORMATION
# ============================================================

# print("Step:", steps, "Reward:", reward)
# Displays the current step number and the reward received.
#
# This helps observe the agent's performance
# during the testing episode.


# ============================================================
# 26. LIMITING TESTING STEPS
# ============================================================

# if steps >= 200:
# Checks whether the number of testing steps has reached 200.


#     break
# Stops the testing loop after 200 steps.
# This prevents the agent from running for too long
# if it does not successfully complete the task.


# ============================================================
# 27. DISPLAYING TOTAL REWARD AND STEPS
# ============================================================

# print("\nTotal Reward:", total_reward)
# Displays the total reward collected during testing.
# A higher reward generally indicates better performance.


# print("Total Steps:", steps)
# Displays the total number of actions performed
# during the testing episode.


# ============================================================
# 28. DISPLAYING THE LEARNED Q-TABLE
# ============================================================

# print("\nLearned Q-Table:")
# Displays the heading for the learned Q-Table.


# print(q_table)
# Displays all learned Q-values.
#
# Rows represent states.
# Columns represent actions.
#
# Higher Q-values indicate actions with better
# estimated future rewards for the corresponding state.


# ============================================================
# 29. CLOSING THE ENVIRONMENT
# ============================================================

# env.close()
# Closes the Taxi environment.
# It releases the resources used by the environment.
#
# Closing the environment is good programming practice.


# ============================================================
# 30. UNDERSTANDING THE COMPLETE OUTPUT
# ============================================================

# State Space:
# Displays the total number of possible Taxi states.
# For the standard Taxi-v3 environment, it is 500.

# Action Space:
# Displays the total number of possible actions.
# For the standard Taxi-v3 environment, it is 6.

# Testing Trained Taxi Agent:
# Indicates that the testing phase has started.

# Step and Reward:
# Displays each action number and the reward received.

# Total Reward:
# Displays the sum of rewards collected during testing.

# Total Steps:
# Displays the number of actions performed during testing.

# Learned Q-Table:
# Displays the Q-values learned by the agent.


# ============================================================
# 31. IMPORTANT TAXI ENVIRONMENT CONCEPTS FOR VIVA
# ============================================================

# Q1. What is Gymnasium?
# Gymnasium is a Python library that provides
# standard environments for developing and testing
# reinforcement learning algorithms.

# Q2. What is the Taxi environment?
# It is a reinforcement learning environment
# in which a taxi must pick up a passenger
# and drop the passenger at the correct destination.

# Q3. What is the objective of the Taxi problem?
# The objective is to pick up the passenger
# and transport the passenger to the correct destination
# using fewer actions and obtaining a good total reward.

# Q4. What is the state space?
# State space represents all possible situations
# of the taxi environment.
# In the standard Taxi environment, there are 500 states.

# Q5. What is the action space?
# Action space represents all possible actions.
# The standard Taxi environment has 6 actions.

# Q6. What are the six Taxi actions?
# 0 = Move South.
# 1 = Move North.
# 2 = Move East.
# 3 = Move West.
# 4 = Pickup passenger.
# 5 = Drop off passenger.

# Q7. What is Q-Learning?
# Q-Learning is a model-free reinforcement learning algorithm.
# It learns the best action for each state
# by updating Q-values using rewards and future rewards.

# Q8. What is a Q-Table?
# A Q-Table stores the expected future reward
# for every state and action combination.

# Q9. What is the size of the Q-Table?
# For the standard Taxi environment:
# 500 states x 6 actions.
# Therefore, the Q-Table size is 500 x 6.

# Q10. What is the learning rate?
# It controls how much new information affects the old Q-value.
# Its value in this practical is 0.7.

# Q11. What is the discount factor?
# It determines the importance of future rewards.
# Its value in this practical is 0.9.

# Q12. What is epsilon?
# Epsilon controls the probability of selecting random actions.
# Its initial value is 1.0.

# Q13. What is epsilon decay?
# Epsilon decay gradually reduces the exploration probability
# after every training episode.

# Q14. What is minimum epsilon?
# It is the lowest exploration probability allowed.
# Its value in this practical is 0.01.

# Q15. What is exploration?
# Exploration means selecting random actions
# to discover better ways of completing the task.

# Q16. What is exploitation?
# Exploitation means selecting the action
# with the highest Q-value.

# Q17. What is the Epsilon-Greedy strategy?
# It selects random actions with probability epsilon.
# Otherwise, it selects the action with the highest Q-value.

# Q18. What is an episode?
# An episode is one complete training attempt
# in the Taxi environment.

# Q19. What is a reward?
# A reward is feedback received by the agent
# after performing an action.

# Q20. What is a model-free algorithm?
# It learns through interaction with the environment
# without requiring a complete model of the environment.

# Q21. What is terminated?
# It indicates that the task has reached a terminal state.
# For example, the passenger has been correctly dropped off.

# Q22. What is truncated?
# It indicates that an episode has stopped
# because of an external limit, such as a time limit.

# Q23. What is env.reset()?
# It resets the environment and returns
# the initial state and additional information.

# Q24. What is env.step(action)?
# It performs an action and returns the next state,
# reward, termination status, truncation status,
# and additional information.

# Q25. What is np.argmax()?
# It returns the index of the maximum value.
# Here, it selects the action with the highest Q-value.

# Q26. What is env.action_space.sample()?
# It randomly selects an action from the available actions.

# Q27. What is the difference between training and testing?
# Training allows the agent to learn through experience.
# Testing checks the agent's performance using the learned Q-Table.

# Q28. What are the applications of Q-Learning?
# Robot navigation.
# Autonomous vehicles.
# Game playing.
# Resource management.
# Traffic control.
# Decision-making systems.


# ============================================================
# 32. CONCLUSION
# ============================================================

# In this practical, the Q-Learning algorithm is implemented
# using the Taxi environment provided by Gymnasium.

# First, the Taxi environment is created.
# The state space and action space are identified.
# A Q-Table is initialized with zeros.

# The agent is trained for 10,000 episodes.
# The Epsilon-Greedy strategy is used for action selection.
# Epsilon gradually decreases during training.

# The Q-Table is updated using the Q-Learning formula.
# After training, the agent is tested using the learned Q-values.

# Finally, the total reward, number of steps,
# and learned Q-Table are displayed.

# Therefore, Q-Learning helps the taxi agent learn
# how to pick up and drop off passengers
# using rewards and penalties.
