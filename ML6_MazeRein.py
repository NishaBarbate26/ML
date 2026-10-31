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
# EXPLANATION LINE BY LINE
# ============================================================

# import numpy as np
# Imports NumPy for numerical calculations.

# import matplotlib.pyplot as plt
# Imports Matplotlib for displaying the maze.

# maze
# Defines the maze using a two-dimensional list.
# 0 represents an empty path.
# 1 represents an obstacle.

# rows and cols
# Store the number of rows and columns.

# start
# Defines the starting position of the agent.

# goal
# Defines the destination of the agent.

# actions
# Defines four possible movements: Up, Down, Left and Right.

# np.zeros()
# Creates a Q-table with initially zero values.

# learning_rate
# Controls how quickly the agent learns.

# discount_factor
# Determines the importance of future rewards.

# epsilon
# Controls random exploration.

# episodes
# Specifies how many times the agent is trained.

# get_state()
# Converts the agent's position into a numerical state.

# take_action()
# Moves the agent according to the selected action.

# if action == 0
# Moves the agent upward.

# elif action == 1
# Moves the agent downward.

# elif action == 2
# Moves the agent left.

# elif action == 3
# Moves the agent right.

# Boundary condition
# Prevents the agent from moving outside the maze.

# Obstacle condition
# Prevents the agent from entering blocked cells.

# Reward 100
# Gives a positive reward when the agent reaches the goal.

# Reward -1
# Gives a small negative reward for each normal movement.

# Reward -10
# Penalizes invalid movements and obstacles.

# for episode in range(episodes)
# Repeats the training process.

# position = start
# Resets the agent to its starting position.

# np.random.random()
# Generates a random number to decide exploration.

# np.random.choice()
# Selects a random action.

# np.argmax()
# Selects the action with the highest Q-value.

# q_table update
# Updates the Q-value using the Q-Learning formula.

# position = new_position
# Updates the agent's current position.

# if done
# Stops the episode when the goal is reached.

# Testing the trained agent
# Uses the learned Q-table to find a path.

# path.append()
# Stores every position visited by the agent.

# np.array()
# Converts the maze into a NumPy array.

# plt.imshow()
# Displays the maze.

# plt.plot()
# Draws the agent's learned path.

# plt.scatter()
# Marks the starting and goal positions.

# plt.show()
# Displays the final maze visualization.