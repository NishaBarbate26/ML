# AIM: Implement Reinforcement Learning using a Maze Environment.

# Installation commands:
# Windows:
# python -m pip install numpy matplotlib
# python practical_maze.py
#
# Ubuntu/Linux:
# pip install numpy matplotlib
# python3 -m pip install numpy matplotlib
# python3 practical_maze.py

import numpy as np
import matplotlib.pyplot as plt

maze = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1],
    [0, 0, 0, 0, 0],
    [1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0]
]

rows = 5
cols = 5

start = (0, 0)
goal = (4, 4)

actions = [0, 1, 2, 3]

# 0 = Up, 1 = Down, 2 = Left, 3 = Right

q_table = np.zeros((rows * cols, len(actions)))

learning_rate = 0.8
discount_factor = 0.9
epsilon = 0.2
episodes = 1000

def get_state(position):
    return position[0] * cols + position[1]

def take_action(position, action):

    row, col = position

    if action == 0:
        row -= 1
    elif action == 1:
        row += 1
    elif action == 2:
        col -= 1
    elif action == 3:
        col += 1

    if row < 0 or row >= rows or col < 0 or col >= cols:
        return position, -10, False

    if maze[row][col] == 1:
        return position, -10, False

    new_position = (row, col)

    if new_position == goal:
        return new_position, 100, True

    return new_position, -1, False

# Training the agent

for episode in range(episodes):

    position = start

    for step in range(100):

        state = get_state(position)

        if np.random.random() < epsilon:
            action = np.random.choice(actions)
        else:
            action = np.argmax(q_table[state])

        new_position, reward, done = take_action(position, action)

        next_state = get_state(new_position)

        q_table[state, action] = q_table[state, action] + learning_rate * (
            reward + discount_factor * np.max(q_table[next_state])
            - q_table[state, action]
        )

        position = new_position

        if done:
            break

# Testing the trained agent

position = start
path = [position]

for step in range(100):

    state = get_state(position)

    action = np.argmax(q_table[state])

    new_position, reward, done = take_action(position, action)

    if new_position == position:
        print("Agent is stuck. Try training again.")
        break

    position = new_position
    path.append(position)

    if done:
        print("Goal reached successfully!")
        break

print("\nLearned Path:")

for point in path:
    print(point)

print("\nQ-Table:")
print(q_table)

# Display maze and learned path

display_maze = np.array(maze, dtype=float)

display_maze[start] = 0.5
display_maze[goal] = 0.5

plt.figure(figsize=(6, 6))

plt.imshow(display_maze, cmap="gray_r")

path_rows = [point[0] for point in path]
path_cols = [point[1] for point in path]

plt.plot(path_cols, path_rows, "ro-", label="Agent Path")

plt.scatter(start[1], start[0], color="green", s=150, label="Start")

plt.scatter(goal[1], goal[0], color="blue", s=150, label="Goal")

plt.title("Q-Learning Maze Environment")

plt.xticks(range(cols))
plt.yticks(range(rows))

plt.grid()

plt.legend()

plt.show()

# ============================================================
# AIM:
# Implement Q-Learning Algorithm to Solve a Maze Problem
# Using Reinforcement Learning.
# ============================================================


# ============================================================
# 1. INSTALLATION AND EXECUTION COMMANDS
# ============================================================

# Windows:
# python -m pip install numpy matplotlib
# python practical_qlearning.py

# Ubuntu/Linux:
# pip install numpy matplotlib
# python3 -m pip install numpy matplotlib
# python3 practical_qlearning.py


# ============================================================
# 2. IMPORTING LIBRARIES
# ============================================================

# import numpy as np
# Imports NumPy for numerical calculations and array operations.
# np is the short name used for NumPy.
# It is used to create the Q-Table and perform calculations.

# import matplotlib.pyplot as plt
# Imports Matplotlib for displaying the maze and learned path.
# plt is the short name used for pyplot.


# ============================================================
# 3. CREATING THE MAZE ENVIRONMENT
# ============================================================

# maze = [
#     [0, 0, 0, 0, 0],
#     [0, 1, 1, 0, 1],
#     [0, 0, 0, 0, 0],
#     [1, 0, 1, 1, 0],
#     [0, 0, 0, 0, 0]
# ]

# Creates a 5 x 5 maze using a two-dimensional list.
#
# 0 represents an empty cell where the agent can move.
# 1 represents a wall or obstacle where the agent cannot move.
#
# The agent must move through empty cells to reach the goal.
#
# Example:
# maze[0][0] = 0 means the starting cell is empty.
# maze[1][1] = 1 means that cell contains an obstacle.


# ============================================================
# 4. DEFINING MAZE DIMENSIONS
# ============================================================

# rows = 5
# Specifies the number of rows in the maze.

# cols = 5
# Specifies the number of columns in the maze.

# The maze contains:
# 5 rows x 5 columns = 25 cells.


# ============================================================
# 5. DEFINING START AND GOAL POSITIONS
# ============================================================

# start = (0, 0)
# Defines the starting position of the agent.
# The agent starts at row 0 and column 0.
# It is the top-left corner of the maze.

# goal = (4, 4)
# Defines the goal position.
# The agent must reach row 4 and column 4.
# It is the bottom-right corner of the maze.


# ============================================================
# 6. DEFINING POSSIBLE ACTIONS
# ============================================================

# actions = [0, 1, 2, 3]
# Defines the four possible movements of the agent.

# 0 = Up
# 1 = Down
# 2 = Left
# 3 = Right

# The agent selects one of these actions
# to move through the maze.


# ============================================================
# 7. CREATING THE Q-TABLE
# ============================================================

# q_table = np.zeros((rows * cols, len(actions)))
# Creates a Q-Table filled with zeros.
#
# rows * cols = 5 * 5 = 25 states.
# len(actions) = 4 possible actions.
#
# Therefore, the Q-Table has 25 rows and 4 columns.
#
# Each row represents a state or maze position.
# Each column represents an action.
#
# Initially, all Q-values are zero.
#
# The Q-Table stores the expected future reward
# for performing an action in a particular state.


# ============================================================
# 8. DEFINING Q-LEARNING PARAMETERS
# ============================================================

# learning_rate = 0.8
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
# A value close to 1 means future rewards are important.
# A value close to 0 means immediate rewards are more important.


# epsilon = 0.2
# Defines the exploration probability.
#
# It is used in the Epsilon-Greedy strategy.
#
# 0.2 means:
# 20% probability of selecting a random action.
# 80% probability of selecting the action
# with the highest Q-value.


# episodes = 1000
# Defines the number of training episodes.
#
# One episode means one complete attempt by the agent
# to reach the goal or complete its allowed steps.
#
# The agent gets 1000 training attempts.


# ============================================================
# 9. DEFINING THE GET_STATE FUNCTION
# ============================================================

# def get_state(position):
# Defines a function to convert a maze position
# into a single numerical state number.


# return position[0] * cols + position[1]
# Converts the row and column position into a state number.
#
# position[0] represents the row.
# position[1] represents the column.
# cols represents the total number of columns.
#
# Formula:
# State = (Row x Number of Columns) + Column
#
# Example:
# Position (0, 0) gives state 0.
# Position (0, 1) gives state 1.
# Position (1, 0) gives state 5.
# Position (4, 4) gives state 24.
#
# Therefore, all 25 maze positions are represented
# using state numbers from 0 to 24.


# ============================================================
# 10. DEFINING THE TAKE_ACTION FUNCTION
# ============================================================

# def take_action(position, action):
# Defines a function that performs the selected action.
#
# position represents the current position of the agent.
# action represents the movement selected by the agent.
#
# The function returns:
# New position, reward, and done status.


# ------------------------------------------------------------
# 10.1 GETTING THE CURRENT POSITION
# ------------------------------------------------------------

# row, col = position
# Separates the current position into row and column values.
#
# Example:
# If position = (2, 3),
# row = 2 and col = 3.


# ------------------------------------------------------------
# 10.2 MOVING UP
# ------------------------------------------------------------

# if action == 0:
# Checks whether the selected action is Up.

#     row -= 1
# Decreases the row number by 1.
# The agent moves one cell upward.


# ------------------------------------------------------------
# 10.3 MOVING DOWN
# ------------------------------------------------------------

# elif action == 1:
# Checks whether the selected action is Down.

#     row += 1
# Increases the row number by 1.
# The agent moves one cell downward.


# ------------------------------------------------------------
# 10.4 MOVING LEFT
# ------------------------------------------------------------

# elif action == 2:
# Checks whether the selected action is Left.

#     col -= 1
# Decreases the column number by 1.
# The agent moves one cell to the left.


# ------------------------------------------------------------
# 10.5 MOVING RIGHT
# ------------------------------------------------------------

# elif action == 3:
# Checks whether the selected action is Right.

#     col += 1
# Increases the column number by 1.
# The agent moves one cell to the right.


# ============================================================
# 11. CHECKING MAZE BOUNDARIES
# ============================================================

# if row < 0 or row >= rows or col < 0 or col >= cols:
# Checks whether the agent is trying to move outside the maze.
#
# row < 0 checks movement above the first row.
# row >= rows checks movement below the last row.
# col < 0 checks movement before the first column.
# col >= cols checks movement beyond the last column.


#     return position, -10, False
# If the agent tries to move outside the maze:
#
# position:
# The agent remains at its current position.
#
# -10:
# The agent receives a penalty of -10.
#
# False:
# The goal has not been reached.
#
# The agent is discouraged from moving outside the maze.


# ============================================================
# 12. CHECKING OBSTACLES
# ============================================================

# if maze[row][col] == 1:
# Checks whether the next cell contains an obstacle.
#
# maze[row][col] accesses the selected maze cell.
# 1 represents a wall.


#     return position, -10, False
# If the next cell contains a wall:
#
# The agent remains in the current position.
# It receives a penalty of -10.
# The episode does not finish because the goal is not reached.


# ============================================================
# 13. UPDATING THE NEW POSITION
# ============================================================

# new_position = (row, col)
# Creates the new position after a valid movement.
# It stores the updated row and column values.


# ============================================================
# 14. CHECKING WHETHER THE GOAL IS REACHED
# ============================================================

# if new_position == goal:
# Checks whether the agent has reached the goal position.


#     return new_position, 100, True
# If the goal is reached:
#
# new_position:
# Returns the goal position.
#
# 100:
# Gives the agent a positive reward of 100.
#
# True:
# Indicates that the episode has successfully finished.
#
# The agent is encouraged to reach the goal.


# ============================================================
# 15. NORMAL MOVEMENT REWARD
# ============================================================

# return new_position, -1, False
# If the movement is valid but the goal is not reached:
#
# The agent moves to the new position.
# It receives a small penalty of -1.
# The episode continues.
#
# This encourages the agent to find a shorter path
# because every normal movement has a small cost.


# ============================================================
# 16. TRAINING THE Q-LEARNING AGENT
# ============================================================

# for episode in range(episodes):
# Starts the training loop.
# It repeats the learning process 1000 times.
# Each repetition is called an episode.


# position = start
# Resets the agent to the starting position
# at the beginning of every episode.


# for step in range(100):
# Allows the agent to perform a maximum of 100 steps
# in each episode.
# This prevents the agent from moving endlessly.


# ============================================================
# 17. GETTING THE CURRENT STATE
# ============================================================

# state = get_state(position)
# Converts the current row and column position
# into a single state number.
# This state number is used to access the Q-Table.


# ============================================================
# 18. SELECTING AN ACTION USING EPSILON-GREEDY
# ============================================================

# if np.random.random() < epsilon:
# Generates a random number between 0 and 1.
# Checks whether it is less than epsilon.
#
# If the condition is true, the agent explores
# by selecting a random action.


#     action = np.random.choice(actions)
# Selects a random action from Up, Down, Left, and Right.
# This is called exploration.


# else:
# Executes when the random number is not less than epsilon.
# The agent uses its existing Q-Table knowledge.


#     action = np.argmax(q_table[state])
# Selects the action having the highest Q-value
# in the current state.
#
# np.argmax() returns the index of the maximum value.
# This is called exploitation.
#
# Exploration means trying new actions.
# Exploitation means using the best-known action.


# ============================================================
# 19. PERFORMING THE SELECTED ACTION
# ============================================================

# new_position, reward, done = take_action(position, action)
# Calls the take_action() function.
#
# new_position stores the next position.
# reward stores the reward or penalty received.
# done indicates whether the goal is reached.


# ============================================================
# 20. GETTING THE NEXT STATE
# ============================================================

# next_state = get_state(new_position)
# Converts the new position into a numerical state number.
# This next state is used to update the Q-Table.


# ============================================================
# 21. UPDATING THE Q-TABLE
# ============================================================

# q_table[state, action] = q_table[state, action] + learning_rate * (
#     reward + discount_factor * np.max(q_table[next_state])
#     - q_table[state, action]
# )

# Updates the Q-value for the current state and action.
#
# This is the main Q-Learning update formula.
#
# Q-Learning Formula:
#
# Q(s,a) = Q(s,a) + alpha [
#     reward + gamma * max Q(next_state, next_action)
#     - Q(s,a)
# ]
#
# Here:
#
# Q(s,a):
# Current Q-value for the selected state and action.
#
# alpha:
# Learning rate.
# In this program, alpha = 0.8.
#
# reward:
# Reward received after performing the action.
#
# gamma:
# Discount factor.
# In this program, gamma = 0.9.
#
# max Q(next_state, next_action):
# Highest Q-value available in the next state.
#
# The formula updates the old Q-value
# using the reward and expected future reward.
#
# This helps the agent learn which actions
# lead to better future results.


# ============================================================
# 22. MOVING TO THE NEXT POSITION
# ============================================================

# position = new_position
# Updates the current position of the agent.
# The agent now moves from the old position to the new position.


# ============================================================
# 23. CHECKING WHETHER TRAINING EPISODE IS COMPLETE
# ============================================================

# if done:
# Checks whether the goal has been reached.


#     break
# Stops the current episode if the goal is reached.
# The next training episode then starts from the start position.


# ============================================================
# 24. TESTING THE TRAINED AGENT
# ============================================================

# position = start
# Resets the agent to the starting position for testing.


# path = [position]
# Creates a list to store the path followed by the agent.
# Initially, the list contains only the starting position.


# for step in range(100):
# Allows the trained agent to perform a maximum of 100 steps.
# This prevents an endless movement loop during testing.


# ============================================================
# 25. GETTING THE CURRENT TESTING STATE
# ============================================================

# state = get_state(position)
# Converts the current position into its state number.


# ============================================================
# 26. SELECTING THE BEST ACTION
# ============================================================

# action = np.argmax(q_table[state])
# Selects the action with the highest Q-value.
# The trained agent uses its learned knowledge.
#
# Unlike training, this testing step does not use
# random exploration.
# It follows the best-known action in the Q-Table.


# ============================================================
# 27. PERFORMING THE TESTING ACTION
# ============================================================

# new_position, reward, done = take_action(position, action)
# Performs the selected movement.
# Returns the new position, reward, and goal status.


# ============================================================
# 28. CHECKING WHETHER THE AGENT IS STUCK
# ============================================================

# if new_position == position:
# Checks whether the agent's position has not changed.
#
# This may happen if the selected action tries to move
# into a wall or outside the maze.


#     print("Agent is stuck. Try training again.")
# Displays a message if the agent cannot move.


#     break
# Stops the testing loop when the agent is stuck.


# ============================================================
# 29. STORING THE LEARNED PATH
# ============================================================

# position = new_position
# Updates the current position after a valid movement.


# path.append(position)
# Adds the new position to the path list.
# This stores every position visited by the agent.


# ============================================================
# 30. CHECKING WHETHER THE GOAL IS REACHED
# ============================================================

# if done:
# Checks whether the agent has reached the goal.


#     print("Goal reached successfully!")
# Displays a success message when the goal is reached.


#     break
# Stops testing because the goal has been reached.


# ============================================================
# 31. DISPLAYING THE LEARNED PATH
# ============================================================

# print("\nLearned Path:")
# Displays the heading for the learned path.


# for point in path:
# Loops through every position stored in the path list.


#     print(point)
# Displays each position visited by the agent.
#
# Example:
# (0, 0)
# (0, 1)
# (0, 2)
#
# Each pair represents (row, column).


# ============================================================
# 32. DISPLAYING THE Q-TABLE
# ============================================================

# print("\nQ-Table:")
# Displays the Q-Table heading.


# print(q_table)
# Displays the learned Q-values for all states and actions.
#
# Each row represents one maze state.
# Each column represents one action.
#
# Column 0 = Up.
# Column 1 = Down.
# Column 2 = Left.
# Column 3 = Right.
#
# Higher Q-values indicate actions with better
# estimated future rewards.


# ============================================================
# 33. PREPARING THE MAZE FOR DISPLAY
# ============================================================

# display_maze = np.array(maze, dtype=float)
# Converts the maze list into a NumPy array.
# dtype=float converts the values into floating-point numbers.
# This array is used for visualization.


# display_maze[start] = 0.5
# Changes the starting cell value to 0.5.
# It is used to display the start position differently.


# display_maze[goal] = 0.5
# Changes the goal cell value to 0.5.
# It is used to display the goal position differently.


# ============================================================
# 34. CREATING THE MAZE GRAPH
# ============================================================

# plt.figure(figsize=(6, 6))
# Creates a square figure with width 6 inches
# and height 6 inches.


# plt.imshow(display_maze, cmap="gray_r")
# Displays the maze as an image.
#
# display_maze contains the maze values.
# cmap="gray_r" uses a reversed grayscale color map.
#
# Different maze values are displayed using different shades.
# Walls, empty cells, and specially marked cells
# can be visually distinguished.


# ============================================================
# 35. EXTRACTING PATH ROWS AND COLUMNS
# ============================================================

# path_rows = [point[0] for point in path]
# Extracts the row number from every position in the path.
# Stores all row values in path_rows.


# path_cols = [point[1] for point in path]
# Extracts the column number from every position in the path.
# Stores all column values in path_cols.


# ============================================================
# 36. DRAWING THE AGENT'S PATH
# ============================================================

# plt.plot(path_cols, path_rows, "ro-", label="Agent Path")
# Draws the path followed by the agent.
#
# path_cols represents horizontal positions.
# path_rows represents vertical positions.
#
# "r" means red color.
# "o" means circular markers.
# "-" connects the points using lines.
#
# label="Agent Path" gives a name to the plotted path.


# ============================================================
# 37. DISPLAYING THE START POSITION
# ============================================================

# plt.scatter(start[1], start[0], color="green",
#             s=150, label="Start")
#
# Displays the starting position as a green point.
#
# start[1] represents the column.
# start[0] represents the row.
# color="green" sets the marker color.
# s=150 sets the marker size.
# label="Start" adds its name to the legend.


# ============================================================
# 38. DISPLAYING THE GOAL POSITION
# ============================================================

# plt.scatter(goal[1], goal[0], color="blue",
#             s=150, label="Goal")
#
# Displays the goal position as a blue point.
#
# goal[1] represents the column.
# goal[0] represents the row.
# color="blue" sets the marker color.
# s=150 sets the marker size.
# label="Goal" adds its name to the legend.


# ============================================================
# 39. ADDING GRAPH DETAILS
# ============================================================

# plt.title("Q-Learning Maze Environment")
# Sets the title of the maze graph.


# plt.xticks(range(cols))
# Displays column numbers on the horizontal axis.
# The values range from 0 to 4.


# plt.yticks(range(rows))
# Displays row numbers on the vertical axis.
# The values range from 0 to 4.


# plt.grid()
# Displays grid lines to separate the maze cells.


# plt.legend()
# Displays the legend for the agent path, start, and goal.


# plt.show()
# Displays the final maze and learned path graph.


# ============================================================
# 40. UNDERSTANDING THE COMPLETE OUTPUT
# ============================================================

# Goal reached successfully!
# Indicates that the agent reached the goal position.

# Learned Path:
# Displays all the positions visited by the agent
# during testing.

# Q-Table:
# Displays the learned Q-values for every state and action.

# Maze Graph:
# Displays the maze, start position, goal position,
# and the path followed by the trained agent.


# ============================================================
# 41. IMPORTANT REINFORCEMENT LEARNING CONCEPTS FOR VIVA
# ============================================================

# Q1. What is Reinforcement Learning?
# Reinforcement Learning is a type of machine learning
# in which an agent learns by interacting with an environment
# and receiving rewards or penalties.

# Q2. What is Q-Learning?
# Q-Learning is a model-free reinforcement learning algorithm.
# It learns the best action to take in each state
# by updating Q-values.

# Q3. What is an agent?
# An agent is the learner or decision-maker.
# In this practical, the agent is the object
# that moves through the maze.

# Q4. What is an environment?
# An environment is the place where the agent operates.
# In this practical, the maze is the environment.

# Q5. What is a state?
# A state represents the current position of the agent.
# Each maze position is converted into a number from 0 to 24.

# Q6. What is an action?
# An action is a movement selected by the agent.
# The four actions are Up, Down, Left, and Right.

# Q7. What is a reward?
# A reward is feedback received by the agent
# after performing an action.
# Positive rewards encourage good actions.
# Negative rewards discourage unwanted actions.

# Q8. What is a Q-Table?
# A Q-Table stores the expected future reward
# for each state and action combination.

# Q9. What is the size of the Q-Table in this practical?
# The maze has 25 states and 4 actions.
# Therefore, the Q-Table size is 25 x 4.

# Q10. What is the learning rate?
# Learning rate controls how much new information
# affects the existing Q-value.
# Its value in this practical is 0.8.

# Q11. What is the discount factor?
# Discount factor determines the importance of future rewards.
# Its value in this practical is 0.9.

# Q12. What is epsilon?
# Epsilon controls the probability of random exploration.
# Its value in this practical is 0.2.

# Q13. What is exploration?
# Exploration means trying random actions
# to discover new paths and rewards.

# Q14. What is exploitation?
# Exploitation means selecting the action
# with the highest known Q-value.

# Q15. What is the Epsilon-Greedy strategy?
# It selects random actions with probability epsilon.
# Otherwise, it selects the action with the highest Q-value.

# Q16. What is an episode?
# An episode is one training attempt by the agent.
# It begins at the start position and ends when the goal
# is reached or the step limit is reached.

# Q17. What is the purpose of the reward +100?
# It encourages the agent to reach the goal.

# Q18. What is the purpose of the penalty -10?
# It discourages the agent from hitting walls
# or moving outside the maze.

# Q19. What is the purpose of the reward -1?
# It discourages unnecessary movements
# and encourages shorter paths.

# Q20. What is the maximum number of training episodes?
# 1000 episodes.

# Q21. What is the maximum number of steps in each episode?
# 100 steps.

# Q22. What is a model-free algorithm?
# A model-free algorithm learns through interaction
# without requiring a complete model of the environment.
# Q-Learning is model-free.

# Q23. What is the Q-Learning update equation?
# Q(s,a) = Q(s,a) + alpha [
#     reward + gamma * max Q(next_state, next_action)
#     - Q(s,a)
# ]

# Q24. What does np.argmax() do?
# It returns the index of the maximum value.
# Here, it selects the action with the highest Q-value.

# Q25. What does np.random.choice() do?
# It randomly selects an action from the available actions.

# Q26. What is the difference between supervised learning
# and reinforcement learning?
# Supervised learning learns from labelled examples.
# Reinforcement learning learns through rewards and penalties.

# Q27. What are the applications of Q-Learning?
# Robot navigation.
# Maze solving.
# Game playing.
# Traffic signal control.
# Autonomous systems.
# Resource management.


# ============================================================
# 42. CONCLUSION
# ============================================================

# In this practical, Q-Learning is implemented
# using a maze environment.

# First, a maze is created with empty cells and obstacles.
# The start and goal positions are defined.
# Four possible actions are provided to the agent.

# A Q-Table is initialized with zeros.
# The agent is trained for 1000 episodes
# using the Epsilon-Greedy strategy.

# The agent receives rewards for reaching the goal
# and penalties for hitting obstacles or taking extra steps.

# The Q-Table is updated using the Q-Learning formula.
# After training, the agent uses the learned Q-values
# to find a path to the goal.

# Finally, the learned path and Q-Table are displayed,
# and Matplotlib is used to visualize the maze.

# Therefore, Q-Learning helps an agent learn
# an optimal or better path through an environment
# using rewards and penalties.

