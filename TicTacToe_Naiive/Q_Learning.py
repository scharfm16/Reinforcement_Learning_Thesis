import numpy as np
import operator

#HYPERPARAMETERS IN Q-LEARNING. RUN CODE IN TEST.PY
#Hyperparameters
num_runs = 20
num_episodes = 100000
qfunction_checkpoint = 20000
gameplay_checkpoint = 10000
display_games = False
display_graphs = True
testing_iter = 1000
testing_eps = 0
gamma = 1
agent_eps = 1
agent_alpha = .1
env_eps = 1
env_alpha = .1
load_bots = False
save_bots = False

#Retrieving the legal actions of the board
def legal_actions(board):
	actions = []
	for i in range(3):
		for j in range(3):
			actions.append(board[i][j])
	legal_actions = [x == 0 for x in actions]
	return legal_actions


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
	legal_action_mask = legal_actions(board)
	Q_next = {}

	for next_action in range(9):

		if legal_action_mask[next_action]:
			# post_state = state + (int(player_num*(3 ** next_action)))
			Q_next[next_action] = Q[state][next_action]

	next_action, next_q = max(Q_next.items(), key=operator.itemgetter(1))
	return next_action, next_q

class QLearning(object):
	# Initialize a Qlearning object
	# alpha is the "learning_rate"
	def __init__(self, num_states, num_actions, alpha=0.5, gamma =.99):

		self.Q = np.zeros((num_states, num_actions))

		self.alpha = alpha
		self.num_actions = num_actions
		self.gamma = gamma


    # updates the Q value table
    # with a (state, action, reward, next_state) from one step in the environment
    # done is a bool indicating if the episode terminated at this step

    #I added the calculation of the max next action so that it wouldn't need to be done again at the policy step
	def update(self, state, action, reward, next_state, done, board):
		if not done:
			max_next_action, max_next_Q = optimal_policy(self.Q, next_state, board)
		else:
			max_next_action, max_next_Q = None , 0

		self.Q[state][action] = ((1-self.alpha) * self.Q[state][action])\
							 + self.alpha * (reward + (self.gamma*max_next_Q))

		return max_next_action



# TODO: fill this in
# run the greedy policy (no randomness) on the environment for niter number of times
# and return the fraction of times that it reached the goal
def evaluate_greedy_policy(qlearning, env, niter=100, display=True):
	num_wins = 0
	num_losses = 0
	num_ties = 0
	for i in range(niter):
		done = False
		env.reset()
		reward = 0

		while not done:
			state = discretized_state(env)
			max_next_action, _ = optimal_policy(qlearning.Q, state, env.agent_state)

			legal_action_mask = legal_actions(env.agent_state)
			actions = [action for action in list(range(9)) if legal_action_mask[action]]
			random_action = np.random.choice(actions)
			action = np.random.choice([max_next_action, random_action], p=[1 - testing_eps, testing_eps])

			dummy, reward, done = env.step(action + 1, bot_train=False, display=display)
		if reward == 10:
			num_wins += 1
		elif reward == -10:
			num_losses += 1
		elif reward == 0:
			num_ties += 1
	return (100* num_wins/niter, 100* num_losses/niter, 100*num_ties/niter)