import numpy as np
import operator
import itertools

#HYPERPARAMETERS IN Q-LEARNING. RUN CODE IN TEST.PY

#Hyperparameters
#Number of times that full Q-Learning is run (results across runs are averaged)
num_runs = 1
#Number of episodes that Q-Learning will be run for
num_episodes = 100000
#Multiple of episodes at which qfunction is evaluated
qfunction_checkpoint = 20000
#Multiple of episodes at which gameplay is simulated
gameplay_checkpoint = 10000
#How many episodes to evaluate the policy after training
testing_iter = 1000
#Noise in policy for testing
testing_eps = 0
#Discount rate
gamma = 1
#Noise in policy for training
eps = 1
#Learning rate
alpha = .1
#Whether to load/save bot states in a local file
load_bots = False
save_bots = False
#Display specifications
display_games = False
display_graphs = False

#Retrieving the legal actions of the board
def legal_actions(board):
	actions = np.zeros((9, 9), dtype=bool)
	first_actions = []
	for i in range(3):
		for j in range(3):
			first_actions.append(board[i][j])
	first_legal_actions = [x == 0 for x in first_actions]

	for i in range(9):
		#Checking if first move is legal

		if first_legal_actions[i] == True:

			#Second move can be the same as first
			actions[i] = first_legal_actions
			#We must remove the first move as an option for the second move
			actions[i,i] = False

	return actions


#Implementing the board state representation in base 3
def discretized_state(env):
	state = 0
	place = 0
	for i in range(3):
		for j in range(3):
			state += env.agent_state[i][j] * (3 ** place)
			place += 1
	return int(state)

#Calculating the optimal policy for the agent given its Q-Function and board state
def optimal_policy(Q, state, board):
	#Get the legal pairs of moves at the state
	legal_action_mask = legal_actions(board)

	next_moves = {}
	#Check if there are any legal move pairs
	if np.any(legal_action_mask):
		# 2 or more spots left

		#Identify legal first moves
		legal_first_moves = np.any(legal_action_mask, axis=1)

		#Get the Q-function matrix at this state
		Q_next = np.copy(Q[state])

		#Removing illegal moves from Q-function
		Q_next[~legal_action_mask] = np.nan

		#Iterate over all first moves
		for first_move in range(9):

			#Check if move is legal
			if legal_first_moves[first_move]:

				#Getting response move (asuming opponent acts optimally)
				second_move = np.nanargmin(Q_next[first_move])

				#Storing move and resultant Q to dictionary
				next_moves[(first_move, second_move)] = Q_next[first_move][second_move]

		# Search over the dictionary of possible moves and choose the one with the best Q-value
		max_min_next_moves, next_q = max(next_moves.items(), key=operator.itemgetter(1))

		return max_min_next_moves, next_q

	else:
		#Only one spot left
		first_move = np.argmin(board)

		#Make the first move the only possible move and return the second move and Q-value as degenerate
		second_move = None
		return (first_move, second_move), 0

class Minimax_QLearning(object):
	# Initialize a Qlearning object
	# alpha is the "learning_rate"
	def __init__(self, num_states, num_actions, alpha=0.5, gamma =.99):

		self.Q = np.zeros((num_states, num_actions, num_actions))

		self.alpha = alpha
		self.num_actions = num_actions
		self.gamma = gamma


    # updates the Q value table
    # with a (state, action, reward, next_state) from one step in the environment
    # done is a bool indicating if the episode terminated at this step

	# Make this be the maxmin step
	def update(self, state, move_pair, reward, next_state, done, board):
		if reward == 10:
			move_pair = (move_pair[0],list(range(9)))
			max_min_next_moves, next_Q = None, 0

		elif reward == -10:
			max_min_next_moves, next_Q = None, 0

		else:
			max_min_next_moves, next_Q = optimal_policy(self.Q, next_state, board)

		first_move, second_move = move_pair
		self.Q[state][first_move][second_move] = ((1 - self.alpha) * self.Q[state][first_move][second_move]) \
												 + self.alpha * (reward + (self.gamma * next_Q))

		return max_min_next_moves


if __name__ == '__main__':
	board = np.ones((3, 3))
	board[0][2] = 0
	print(board)
	print(legal_actions(board))