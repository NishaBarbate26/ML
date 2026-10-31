# AIM: Build a Tic-Tac-Toe Game Using Reinforcement Learning.

# Installation commands:
# Windows:
# python -m pip install numpy
# python practical_tictactoe.py
#
# Ubuntu/Linux:
# pip install numpy
# python3 -m pip install numpy
# python3 practical_tictactoe.py

import numpy as np
import random

# Create Q-table

q_table = {}

learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.2

episodes = 50000

# Define winning combinations

winning_positions = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6]
]

# Create empty board

def create_board():
    return [" "] * 9

# Display board

def display_board(board):

    print()

    for i in range(0, 9, 3):
        print(board[i], "|", board[i + 1], "|", board[i + 2])

        if i < 6:
            print("--+---+--")

    print()

# Check winner

def check_winner(board, player):

    for combination in winning_positions:

        if all(board[i] == player for i in combination):
            return True

    return False

# Check draw

def is_draw(board):
    return " " not in board

# Convert board into state

def get_state(board):
    return "".join(board)

# Get available moves

def available_moves(board):
    return [i for i in range(9) if board[i] == " "]

# Get Q-value

def get_q_value(state, action):
    return q_table.get((state, action), 0.0)

# Select action

def choose_action(board, explore=True):

    state = get_state(board)

    moves = available_moves(board)

    if explore and random.random() < epsilon:
        return random.choice(moves)

    values = [get_q_value(state, action) for action in moves]

    best_value = max(values)

    best_actions = [
        action for action, value in zip(moves, values)
        if value == best_value
    ]

    return random.choice(best_actions)

# Update Q-table

def update_q_table(state, action, reward, next_state, next_moves):

    old_value = get_q_value(state, action)

    if next_moves:
        future_value = max(
            get_q_value(next_state, move)
            for move in next_moves
        )
    else:
        future_value = 0

    new_value = old_value + learning_rate * (
        reward + discount_factor * future_value - old_value
    )

    q_table[(state, action)] = new_value

# Train the agent

for episode in range(episodes):

    board = create_board()

    done = False

    while not done:

        # Agent plays as X

        state = get_state(board)

        action = choose_action(board)

        board[action] = "X"

        if check_winner(board, "X"):

            update_q_table(state, action, 1, get_state(board), [])

            break

        if is_draw(board):

            update_q_table(state, action, 0, get_state(board), [])

            break

        # Opponent plays randomly as O

        opponent_action = random.choice(available_moves(board))

        board[opponent_action] = "O"

        if check_winner(board, "O"):

            update_q_table(state, action, -1, get_state(board), [])

            break

        if is_draw(board):

            update_q_table(state, action, 0, get_state(board), [])

            break

        next_state = get_state(board)

        next_moves = available_moves(board)

        update_q_table(
            state,
            action,
            0,
            next_state,
            next_moves
        )

# Test the trained agent

print("\nTic-Tac-Toe Reinforcement Learning Game")

print("You are O and the computer is X.")

board = create_board()

while True:

    display_board(board)

    # Computer move

    action = choose_action(board, explore=False)

    board[action] = "X"

    print("Computer played:", action + 1)

    if check_winner(board, "X"):

        display_board(board)

        print("Computer Wins!")

        break

    if is_draw(board):

        display_board(board)

        print("Match Draw!")

        break

    # Human move

    display_board(board)

    while True:

        try:
            user_move = int(input("Enter your position (1-9): ")) - 1

            if user_move in available_moves(board):
                break

            print("Invalid position. Try again.")

        except ValueError:
            print("Enter a valid number.")

    board[user_move] = "O"

    if check_winner(board, "O"):

        display_board(board)

        print("You Win!")

        break

    if is_draw(board):

        display_board(board)

        print("Match Draw!")

        break


# ============================================================
# EXPLANATION LINE BY LINE
# ============================================================

# import numpy as np
# Imports NumPy for numerical operations.

# import random
# Imports random for selecting moves.

# q_table = {}
# Creates an empty dictionary to store state-action values.

# learning_rate
# Controls how quickly the agent learns.

# discount_factor
# Determines the importance of future rewards.

# epsilon
# Controls random exploration.

# episodes
# Specifies the number of training games.

# winning_positions
# Stores all eight possible winning combinations.

# create_board()
# Creates an empty Tic-Tac-Toe board with nine positions.

# display_board()
# Displays the Tic-Tac-Toe board.

# check_winner()
# Checks whether the given player has won.

# is_draw()
# Checks whether all board positions are occupied.

# get_state()
# Converts the board into a string representing the current state.

# available_moves()
# Returns all empty positions on the board.

# get_q_value()
# Returns the Q-value of a particular state and action.

# choose_action()
# Selects an action using exploration or learned Q-values.

# random.random()
# Generates a random number to decide exploration.

# random.choice()
# Selects a random available move.

# max(values)
# Finds the highest Q-value among available moves.

# update_q_table()
# Updates the Q-value using the Q-Learning formula.

# old_value
# Stores the previous Q-value.

# future_value
# Stores the maximum expected future Q-value.

# new_value
# Calculates the updated Q-value.

# for episode in range(episodes)
# Repeats training for the specified number of games.

# board = create_board()
# Starts every training game with an empty board.

# choose_action()
# Selects the agent's move.

# board[action] = "X"
# Places the agent's mark on the board.

# check_winner()
# Checks whether the agent has won.

# reward = 1
# Gives a positive reward when the agent wins.

# reward = -1
# Gives a negative reward when the agent loses.

# reward = 0
# Gives no reward for a draw or an intermediate move.

# random.choice(available_moves(board))
# Allows the opponent to select a random move.

# next_state
# Stores the board after the opponent's move.

# next_moves
# Stores the available moves in the next state.

# choose_action(explore=False)
# Selects the best learned move without random exploration.

# input()
# Accepts the human player's move.

# int()
# Converts the entered position into an integer.

# display_board()
# Displays the updated game after each move.