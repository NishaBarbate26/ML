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
# AIM:
# Implement Q-Learning Algorithm for Tic-Tac-Toe Game.
# Train the computer to play Tic-Tac-Toe using Reinforcement
# Learning and allow a human to play against the trained agent.
# ============================================================


# ============================================================
# 1. IMPORTING LIBRARIES
# ============================================================

# import numpy as np
# Imports the NumPy library.
# NumPy is generally used for numerical calculations and arrays.
# In this particular program, NumPy is imported but not used.
# Therefore, it can be removed without affecting the program.

# import random
# Imports Python's random module.
# It is used to select random moves during training.
# It is also used when multiple actions have the same Q-value.


# ============================================================
# 2. CREATING THE Q-TABLE
# ============================================================

# q_table = {}
# Creates an empty dictionary called Q-Table.
#
# A Q-Table stores the learned value of each state-action pair.
#
# State:
# Represents the current arrangement of the Tic-Tac-Toe board.
#
# Action:
# Represents the position where the computer can place X.
#
# Unlike previous practicals, this program uses a dictionary
# instead of a NumPy matrix.
#
# The dictionary stores:
# (state, action) : Q-value
#
# Example:
# ("         ", 4) : 0.5
#
# This means that placing X at position 4
# in an empty board has a Q-value of 0.5.


# ============================================================
# 3. DEFINING LEARNING PARAMETERS
# ============================================================

# learning_rate = 0.1
# Defines the learning rate, represented by alpha.
#
# It controls how much new information affects the old Q-value.
#
# A value of 0.1 means that 10% of the calculated learning
# update is applied to the existing Q-value.


# discount_factor = 0.9
# Defines the discount factor, represented by gamma.
#
# It determines the importance of future rewards.
#
# A value of 0.9 means that future rewards are given
# considerable importance.


# epsilon = 0.2
# Defines the exploration probability.
#
# The agent has a 20% probability of selecting a random move.
# Otherwise, it selects a move based on its Q-Table.
#
# Exploration helps the agent discover different moves.
# Exploitation helps the agent use its learned knowledge.


# episodes = 50000
# Defines the total number of training games.
#
# The computer plays 50,000 games against a random opponent.
#
# By playing many games, the computer gets opportunities
# to learn which moves are useful and which moves are harmful.


# ============================================================
# 4. DEFINING WINNING COMBINATIONS
# ============================================================

# winning_positions = [
#     [0, 1, 2],
#     [3, 4, 5],
#     [6, 7, 8],
#     [0, 3, 6],
#     [1, 4, 7],
#     [2, 5, 8],
#     [0, 4, 8],
#     [2, 4, 6]
# ]
#
# Stores all eight possible winning combinations.
#
# A player wins when the same symbol appears
# in any one of these combinations.
#
# The board positions are numbered from 0 to 8.
#
# The eight winning combinations are:
#
# First row:  [0, 1, 2]
# Second row: [3, 4, 5]
# Third row:  [6, 7, 8]
#
# First column:  [0, 3, 6]
# Second column: [1, 4, 7]
# Third column:  [2, 5, 8]
#
# Main diagonal:     [0, 4, 8]
# Other diagonal:    [2, 4, 6]


# ============================================================
# 5. CREATING AN EMPTY BOARD
# ============================================================

# def create_board():
# Defines a function called create_board.
# This function creates a new Tic-Tac-Toe board.


#     return [" "] * 9
# Returns a list containing nine empty spaces.
#
# Each empty space represents one position on the board.
#
# The board initially looks like:
#
#  |  |
# --+---+--
#  |  |
# --+---+--
#  |  |
#
# The board contains nine positions numbered 0 to 8.


# ============================================================
# 6. DISPLAYING THE BOARD
# ============================================================

# def display_board(board):
# Defines a function to display the current board.
# It receives the board as an argument.


#     print()
# Prints an empty line for better readability.


#     for i in range(0, 9, 3):
# Starts a loop from 0 to 8 with a step of 3.
#
# The values of i are:
# 0, 3, 6.
#
# Each value represents the starting position of a row.


#         print(board[i], "|", board[i + 1], "|", board[i + 2])
# Displays three positions of the current row.
#
# When i = 0:
# Displays positions 0, 1, and 2.
#
# When i = 3:
# Displays positions 3, 4, and 5.
#
# When i = 6:
# Displays positions 6, 7, and 8.
#
# The vertical bars separate the three columns.


#         if i < 6:
# Checks whether the current row is not the last row.
#
# This condition is true for i = 0 and i = 3.


#             print("--+---+--")
# Displays a horizontal separator between rows.


#     print()
# Prints an empty line after displaying the board.


# ============================================================
# 7. CHECKING THE WINNER
# ============================================================

# def check_winner(board, player):
# Defines a function to check whether a player has won.
#
# board:
# Represents the current game board.
#
# player:
# Represents the player symbol, either X or O.


#     for combination in winning_positions:
# Checks each of the eight winning combinations.


#         if all(board[i] == player for i in combination):
# Checks whether all three positions in a combination
# contain the same player's symbol.
#
# all() returns True only when every condition is True.
#
# For example:
# If positions 0, 1, and 2 all contain X,
# the computer wins.


#             return True
# Returns True if a winning combination is found.


#     return False
# Returns False if no winning combination is found.


# ============================================================
# 8. CHECKING WHETHER THE GAME IS A DRAW
# ============================================================

# def is_draw(board):
# Defines a function to check whether the game is a draw.


#     return " " not in board
# Checks whether there are any empty positions left.
#
# If no empty spaces remain, the function returns True.
#
# If at least one empty space remains, it returns False.
#
# A draw occurs when the board is full
# and neither player has won.


# ============================================================
# 9. CONVERTING THE BOARD INTO A STATE
# ============================================================

# def get_state(board):
# Defines a function to convert the current board into a state.


#     return "".join(board)
# Joins all nine board positions into a single string.
#
# This string represents the current state of the game.
#
# Example:
# ["X", "O", " ", " ", "X", " ", " ", " ", "O"]
#
# Becomes:
# "XO  X   O"
#
# This string is used as the state key in the Q-Table.


# ============================================================
# 10. GETTING AVAILABLE MOVES
# ============================================================

# def available_moves(board):
# Defines a function to find all empty board positions.


#     return [i for i in range(9) if board[i] == " "]
# Checks all positions from 0 to 8.
#
# If a position contains an empty space,
# its index is added to the returned list.
#
# Example:
# If positions 0, 2, and 5 are empty,
# the function returns:
# [0, 2, 5]
#
# The computer and human can only select available positions.


# ============================================================
# 11. GETTING THE Q-VALUE
# ============================================================

# def get_q_value(state, action):
# Defines a function to get the Q-value of a state-action pair.


#     return q_table.get((state, action), 0.0)
# Searches for the given state-action pair in the Q-Table.
#
# If the pair already exists:
# Returns its stored Q-value.
#
# If the pair does not exist:
# Returns 0.0 as the default value.
#
# This allows the program to handle states
# that have not been visited before.


# ============================================================
# 12. SELECTING AN ACTION
# ============================================================

# def choose_action(board, explore=True):
# Defines a function to select the computer's next move.
#
# board:
# Represents the current game board.
#
# explore:
# Determines whether random exploration is allowed.
#
# During training, explore is True.
# During testing, explore is False.


#     state = get_state(board)
# Converts the current board into a state string.
# This state is used to search the Q-Table.


#     moves = available_moves(board)
# Gets all empty positions where the computer can place X.


#     if explore and random.random() < epsilon:
# Checks whether exploration is enabled
# and the generated random number is less than epsilon.
#
# During training, this condition can be true.
# During testing, explore=False makes this condition false.


#         return random.choice(moves)
# Selects and returns one random available move.
#
# This is called exploration.
# The agent tries a move instead of always selecting
# the move with the highest learned Q-value.


#     values = [get_q_value(state, action) for action in moves]
# Gets the Q-value of every available move.
#
# It creates a list containing the Q-values
# corresponding to the available positions.


#     best_value = max(values)
# Finds the highest Q-value among the available moves.
#
# A higher Q-value indicates a better estimated action.


#     best_actions = [
#         action for action, value in zip(moves, values)
#         if value == best_value
#     ]
# Finds all moves having the highest Q-value.
#
# zip() combines each available move with its Q-value.
#
# If multiple moves have the same highest Q-value,
# all those moves are stored in best_actions.


#     return random.choice(best_actions)
# Randomly selects one move from the best available moves.
#
# This avoids always selecting the first move
# when multiple actions have equal Q-values.
#
# This is called exploitation because the agent
# uses its learned Q-Table.


# ============================================================
# 13. UPDATING THE Q-TABLE
# ============================================================

# def update_q_table(state, action, reward, next_state, next_moves):
# Defines a function to update the Q-value.
#
# state:
# Previous board state.
#
# action:
# Move selected by the computer.
#
# reward:
# Reward or penalty received.
#
# next_state:
# Board state after the move and opponent's response.
#
# next_moves:
# Available actions in the next state.


#     old_value = get_q_value(state, action)
# Gets the existing Q-value for the selected state and action.
#
# If it does not exist, the old value is zero.


#     if next_moves:
# Checks whether any moves are available in the next state.
#
# If the list is not empty, future Q-values can be calculated.


#         future_value = max(
#             get_q_value(next_state, move)
#             for move in next_moves
#         )
# Gets the Q-value of every available next move.
# Finds the maximum value.
#
# This represents the best estimated future value
# from the next state.


#     else:
# Executes when there are no available next moves.


#         future_value = 0
# Sets the future value to zero.
#
# This happens when the game has ended
# because of a win or a draw.


#     new_value = old_value + learning_rate * (
#         reward + discount_factor * future_value - old_value
#     )
# Calculates the updated Q-value.
#
# Q-Learning formula:
#
# Q(s,a) = Q(s,a) + alpha [
#     reward + gamma * max Q(next_state, next_action)
#     - Q(s,a)
# ]
#
# old_value:
# Existing Q-value.
#
# learning_rate:
# Controls the amount of learning.
#
# reward:
# Immediate reward or penalty.
#
# discount_factor:
# Controls the importance of future rewards.
#
# future_value:
# Highest estimated Q-value in the next state.
#
# The formula updates the Q-value using the experience
# received by the computer.


#     q_table[(state, action)] = new_value
# Stores the newly calculated Q-value in the dictionary.
#
# The state and action together form the dictionary key.
# The updated Q-value is stored as its corresponding value.


# ============================================================
# 14. TRAINING THE AGENT
# ============================================================

# for episode in range(episodes):
# Starts the training loop.
#
# The computer plays 50,000 games.
# Each game is one training episode.


#     board = create_board()
# Creates a fresh empty board for the current episode.


#     done = False
# Initializes the game status.
# The training loop continues until a win or draw occurs.
#
# Note:
# In this program, done is initialized but never changed.
# The loop actually ends using break statements.
# Therefore, this variable is unnecessary.


#     while not done:
# Starts the game loop.
# Continues playing until a break statement is executed.


# ============================================================
# 15. COMPUTER PLAYS AS X
# ============================================================

#         state = get_state(board)
# Saves the current board arrangement as a state.
# This state is used for the Q-Table update.


#         action = choose_action(board)
# Selects the computer's move using the Epsilon-Greedy strategy.
#
# Since explore=True by default,
# the computer can explore or exploit.


#         board[action] = "X"
# Places X at the selected position.
# The board is updated with the computer's move.


# ============================================================
# 16. CHECKING WHETHER THE COMPUTER WINS
# ============================================================

#         if check_winner(board, "X"):
# Checks whether the computer has completed
# any of the eight winning combinations.


#             update_q_table(state, action, 1, get_state(board), [])
# Updates the Q-value with reward +1.
#
# A positive reward indicates that the computer has won.
#
# The future moves list is empty because the game has ended.
# Therefore, the future Q-value is zero.


#             break
# Ends the current training game because the computer won.


# ============================================================
# 17. CHECKING WHETHER THE GAME IS A DRAW
# ============================================================

#         if is_draw(board):
# Checks whether all board positions are occupied.


#             update_q_table(state, action, 0, get_state(board), [])
# Updates the Q-value with reward 0.
#
# A draw does not provide a positive or negative reward
# in this implementation.


#             break
# Ends the current training game because of a draw.


# ============================================================
# 18. RANDOM OPPONENT PLAYS AS O
# ============================================================

#         opponent_action = random.choice(available_moves(board))
# Selects a random empty position for the opponent.
#
# The opponent does not use a Q-Table.
# It selects moves randomly.


#         board[opponent_action] = "O"
# Places O at the selected position.
# The board is updated with the opponent's move.


# ============================================================
# 19. CHECKING WHETHER THE OPPONENT WINS
# ============================================================

#         if check_winner(board, "O"):
# Checks whether the opponent has completed
# any winning combination.


#             update_q_table(state, action, -1, get_state(board), [])
# Updates the computer's selected action with reward -1.
#
# A negative reward represents a loss for the computer.
#
# This teaches the agent that the selected action
# was followed by an opponent win.


#             break
# Ends the current training game because the opponent won.


# ============================================================
# 20. CHECKING FOR A DRAW AFTER THE OPPONENT'S MOVE
# ============================================================

#         if is_draw(board):
# Checks whether the board is full after the opponent's move.


#             update_q_table(state, action, 0, get_state(board), [])
# Updates the Q-value with reward zero because the game is drawn.


#             break
# Ends the current training game.


# ============================================================
# 21. UPDATING THE NEXT STATE
# ============================================================

#         next_state = get_state(board)
# Converts the board after the opponent's move into a new state.
#
# This state includes both:
# The computer's latest move.
# The opponent's latest move.


#         next_moves = available_moves(board)
# Finds the remaining empty positions.
# These are the possible actions for the computer's next turn.


#         update_q_table(
#             state,
#             action,
#             0,
#             next_state,
#             next_moves
#         )
# Updates the Q-value for the computer's selected move.
#
# Reward is zero because the game has not ended.
#
# The Q-Table uses the next state's best available Q-value
# to estimate the future reward.
#
# This helps the computer learn from future game situations.


# ============================================================
# 22. TESTING THE TRAINED AGENT
# ============================================================

# print("\nTic-Tac-Toe Reinforcement Learning Game")
# Displays the title of the game.


# print("You are O and the computer is X.")
# Displays the symbols assigned to the computer and human.


# board = create_board()
# Creates a new empty board for testing.
#
# The training process is completed.
# The computer now uses the learned Q-Table.


# ============================================================
# 23. STARTING THE HUMAN-COMPUTER GAME
# ============================================================

# while True:
# Starts an infinite game loop.
# The loop ends when a player wins or the game is drawn.


#     display_board(board)
# Displays the current Tic-Tac-Toe board.


# ============================================================
# 24. COMPUTER'S MOVE
# ============================================================

#     action = choose_action(board, explore=False)
# Selects the computer's move using the learned Q-Table.
#
# explore=False disables random exploration.
# The computer chooses among the best-known available moves.


#     board[action] = "X"
# Places X at the computer's selected position.


#     print("Computer played:", action + 1)
# Displays the computer's selected position.
#
# Internally, positions are numbered from 0 to 8.
# For the user, positions are displayed from 1 to 9.
#
# Adding 1 converts the internal position
# into a human-friendly position number.


# ============================================================
# 25. CHECKING WHETHER THE COMPUTER WINS
# ============================================================

#     if check_winner(board, "X"):
# Checks whether the computer has won.


#         display_board(board)
# Displays the final board after the computer wins.


#         print("Computer Wins!")
# Displays the computer's winning message.


#         break
# Stops the game loop.


# ============================================================
# 26. CHECKING WHETHER THE GAME IS DRAWN
# ============================================================

#     if is_draw(board):
# Checks whether all nine positions are occupied.


#         display_board(board)
# Displays the final board.


#         print("Match Draw!")
# Displays the draw message.


#         break
# Stops the game loop.


# ============================================================
# 27. HUMAN'S TURN
# ============================================================

#     display_board(board)
# Displays the updated board before the human selects a move.


#     while True:
# Starts a loop to accept a valid human move.
# The loop continues until the user enters an available position.


#         try:
# Starts a try block.
# It is used to handle invalid input errors.


#             user_move = int(input("Enter your position (1-9): ")) - 1
# Accepts the human's position as input.
#
# int() converts the input into an integer.
#
# The user enters a position from 1 to 9.
# Subtracting 1 converts it into an index from 0 to 8.
#
# Example:
# User enters 1, program uses index 0.
# User enters 5, program uses index 4.
# User enters 9, program uses index 8.


#             if user_move in available_moves(board):
# Checks whether the selected position is empty.
#
# If the position is available,
# the human can place O there.


#                 break
# Exits the input loop when the move is valid.


#             print("Invalid position. Try again.")
# Displays an error message when the selected position
# is already occupied or outside the valid range.


#         except ValueError:
# Handles an error that occurs when the user enters
# something that cannot be converted into an integer.
#
# Example:
# Entering "hello" causes a ValueError.


#             print("Enter a valid number.")
# Asks the user to enter a valid numerical position.


# ============================================================
# 28. PLACING THE HUMAN'S MOVE
# ============================================================

#     board[user_move] = "O"
# Places O at the human's selected position.
# The board is updated.


# ============================================================
# 29. CHECKING WHETHER THE HUMAN WINS
# ============================================================

#     if check_winner(board, "O"):
# Checks whether the human has completed a winning combination.


#         display_board(board)
# Displays the final board after the human wins.


#         print("You Win!")
# Displays the human's winning message.


#         break
# Ends the game loop.


# ============================================================
# 30. CHECKING FOR A DRAW AFTER THE HUMAN'S MOVE
# ============================================================

#     if is_draw(board):
# Checks whether all board positions are occupied.


#         display_board(board)
# Displays the final board.


#         print("Match Draw!")
# Displays the draw message.


#         break
# Ends the game loop.


# ============================================================
# 31. UNDERSTANDING THE COMPLETE OUTPUT
# ============================================================

# Tic-Tac-Toe Reinforcement Learning Game:
# Displays the game heading.

# You are O and the computer is X:
# Shows which symbol belongs to each player.

# Computer played:
# Displays the position selected by the computer.

# Enter your position (1-9):
# Asks the human to select a board position.

# Invalid position. Try again:
# Appears if the selected position is already occupied
# or is outside the valid range.

# Enter a valid number:
# Appears if the user enters non-numeric input.

# Computer Wins:
# Appears when the computer completes a winning combination.

# You Win:
# Appears when the human completes a winning combination.

# Match Draw:
# Appears when all positions are occupied
# and neither player wins.


# ============================================================
# 32. IMPORTANT VIVA QUESTIONS
# ============================================================

# Q1. What is Reinforcement Learning?
# Reinforcement Learning is a type of machine learning
# in which an agent learns by interacting with an environment
# and receiving rewards or penalties.

# Q2. What is Q-Learning?
# Q-Learning is a model-free reinforcement learning algorithm
# that learns the best action for each state.

# Q3. What is a Q-Table?
# A Q-Table stores Q-values for state-action combinations.

# Q4. What is a state in this practical?
# The current arrangement of X, O, and empty spaces
# on the Tic-Tac-Toe board.

# Q5. What is an action?
# An action is the position where the computer places X.

# Q6. What is a reward?
# A reward is feedback received by the agent.
# Positive reward indicates a good outcome.
# Negative reward indicates an undesirable outcome.

# Q7. What is the reward for winning?
# +1.

# Q8. What is the reward for losing?
# -1.

# Q9. What is the reward for a draw?
# 0.

# Q10. What is the learning rate?
# It controls how much new information affects
# the existing Q-value.
# Its value is 0.1.

# Q11. What is the discount factor?
# It controls the importance of future rewards.
# Its value is 0.9.

# Q12. What is epsilon?
# It controls the probability of selecting random actions.
# Its value is 0.2.

# Q13. What is exploration?
# Exploration means trying random moves
# to discover better actions.

# Q14. What is exploitation?
# Exploitation means selecting moves based on
# the highest learned Q-value.

# Q15. What is the Epsilon-Greedy strategy?
# It selects a random action with probability epsilon.
# Otherwise, it selects the action with the highest Q-value.

# Q16. Why are 50,000 episodes used?
# A large number of episodes gives the computer
# more opportunities to learn from different game situations.

# Q17. Why is a random opponent used?
# The random opponent provides different game situations
# during training.

# Q18. Why is a dictionary used for the Q-Table?
# A dictionary stores state-action pairs as keys
# and their Q-values as values.
# It can store only the state-action combinations
# that have been visited.

# Q19. What is the purpose of check_winner()?
# It checks whether a player has completed
# any of the eight winning combinations.

# Q20. What is the purpose of is_draw()?
# It checks whether the board is full
# and no empty position remains.

# Q21. What is the purpose of get_state()?
# It converts the board into a string
# that can be used as a Q-Table state.

# Q22. What is the purpose of available_moves()?
# It returns all empty board positions.

# Q23. What is the purpose of np.argmax()?
# It returns the index of the maximum value.
# Note: This particular practical does not use np.argmax().

# Q24. What is the purpose of random.choice()?
# It randomly selects one item from a list.

# Q25. What is a model-free algorithm?
# An algorithm that learns through interaction
# without needing a complete model of the environment.

# Q26. What are the applications of Q-Learning?
# Game playing.
# Robot navigation.
# Autonomous systems.
# Resource management.
# Traffic signal control.
# Decision-making systems.


# ============================================================
# 33. CONCLUSION
# ============================================================

# In this practical, the Q-Learning algorithm is implemented
# to develop a Tic-Tac-Toe playing agent.

# First, an empty game board and winning combinations are defined.
# The Q-Table is initialized as an empty dictionary.

# The computer is trained for 50,000 episodes
# against a random opponent.

# During training, the computer uses the Epsilon-Greedy strategy
# to select actions and updates its Q-values
# using the Q-Learning formula.

# After training, a human can play against the computer.
# The computer selects moves using the learned Q-Table.

# Therefore, the practical demonstrates how reinforcement
# learning can be used to train an agent to play a game.
