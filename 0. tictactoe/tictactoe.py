"""
Tic Tac Toe Player
"""

import copy
import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    # Count number of Xs ans Os on a board
    number_of_O = 0
    number_of_X = 0
    for row in range(len(board)):
        for column in range(len(board)):
            if board[row][column] == X:
                number_of_X += 1
            elif board[row][column] == O:
                number_of_O += 1

    # If players made equal turns, X has the next turn
    if number_of_O == number_of_X:
        return X
    return O


def actions(board) -> set:
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possible_actions = set()
    for row in range(len(board)):
        for column in range(len(board)):
            if board[row][column] == EMPTY:
                possible_actions.add((row, column))
    return possible_actions


def result(board, action) -> list:
    """
    Returns the board that results from making move (i, j) on the board.
    """
    # Check if action is valid
    if action not in actions(board):
        raise Exception
    
    # Make deep copy of board
    result_board = copy.deepcopy(board)

    # Make a move
    result_board[action[0]][action[1]] = player(result_board)
    return result_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Check if its a start of a game
    if board == initial_state():
        return None

    first_diagonal = []
    second_diagonal = []

    for row in range(len(board)):
        # Get columns and diagonals
        one_column = []
        for column in range(len(board)):
            one_column.append(board[column][row])
            if column == row:
                first_diagonal.append(board[row][column])
            if column + row == 2:
                second_diagonal.append(board[row][column])
        
        # Search for a winner in a row
        winner_in_a_row = check_dimension(board[row])
        if winner_in_a_row:
            return winner_in_a_row

        # Search for a winner in a column
        winner_in_a_column = check_dimension(one_column)
        if winner_in_a_column:
            return winner_in_a_column
        
    # Search for winner in diagonals
    first_diagonal_winner = check_dimension(first_diagonal)
    second_diagonal_winner = check_dimension(second_diagonal)
    if first_diagonal_winner:
        return first_diagonal_winner
    if second_diagonal_winner:
        return second_diagonal_winner
    
    # Tie case
    return None
    

def check_dimension(dimension):
    """Check one row / column / diagonal"""
    if EMPTY in dimension:
        return None
    if X in dimension and O in dimension:
        return None
    if X in dimension:
        return X
    return O
            

def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # Check if one of players won
    if winner(board):
        return True
    
    # Check if there is empty space left
    if not actions(board):
        return True
    
    # Game still in progress
    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    winner_player = winner(board)
    if winner_player == X:
        return 1
    if winner_player == O:
        return -1
    return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None
    
    best_possible_action = ()
    
    current_player = player(board)
    if current_player == X:
        # Count best value for max-player
        max_value = -2
        for action in actions(board):
            action_value = get_min_value(result(board, action))
            if action_value == 1:
                return action
            if action_value > max_value:
                max_value = action_value
                best_possible_action = action

    if current_player == O:
        # Count best value for min-player
        min_value = 2
        for action in actions(board):
            action_value = get_max_value(result(board, action))
            if action_value == -1:
                return action
            if action_value < min_value:
                min_value = action_value
                best_possible_action = action

    return best_possible_action


def get_max_value(board):
    """Count the action with maximum value possible"""

    # If game is over - count the result
    if terminal(board):
        return utility(board)
    
    # Count the highest possible action
    value = -2
    for action in actions(board):
        value = max(value, get_min_value(result(board, action)))
        if value == 1:
            return value
    return value


def get_min_value(board):
    """Count the action with minimal value possible"""
    
    # If game is over - count the result
    if terminal(board):
        return utility(board)
    
    # Count the lowest possible action
    value = 2
    for action in actions(board):
        value = min(value, get_max_value(result(board, action)))
        if value == -1:
            return value
    return value
