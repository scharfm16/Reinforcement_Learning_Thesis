from tictactoe_env import TicTacToe
import matplotlib.pyplot as plt
from Q_Learning import *
#HYPERPARAMETERS IN Q-LEARNING. RUN CODE IN TEST.PY

def base10_state_to_board(n):
	board = [' ']*9
	for i in range(8, -1, -1):
		if n > 0:
			n, r = divmod(n, 3)
			if r == 0:
				board[i] = ' '
			elif r == 1:
				board[i] = 'X'
			else:
				board[i] = 'O'
	return np.array(board[::-1]).reshape((3,3))


#TODO: Average relavent stats over 20 runs
def main():
	eps = agent_eps
	num_states = 3 ** 9
	num_actions = 9
	returns_list = []
	wins_list = []
	losses_list = []
	ties_list = []
	first_state_Qs = []
	sum_of_10000_returns = 0
	env = TicTacToe()
	qlearning = QLearning(num_states, num_actions, alpha=agent_alpha)

	if load_bots:
		qlearning.Q = np.load("play_first_bot1.npy")
		env.qlearning.Q = np.load("play_second_bot1.npy")

	for i in range(num_episodes+1):
		total_returns = 0
		done = False
		env.reset()
		state = discretized_state(env)

		first_state_Q = np.max(qlearning.Q[state])
		first_state_Qs.append(first_state_Q)

		max_next_action, _ = optimal_policy(qlearning.Q, state, env.agent_state)
		while not done:

			#Epsilon greedy policy
			legal_action_mask = legal_actions(env.agent_state)
			actions = [action for action in list(range(9)) if legal_action_mask[action]]
			random_action = np.random.choice(actions)
			action = np.random.choice([max_next_action, random_action], p=[1 - eps, eps])

			#Step
			dummy, reward, done = env.step(action + 1)

			next_state = discretized_state(env)

			#Q update
			max_next_action = qlearning.update(state, action, reward, next_state, done, env.agent_state)


			state = next_state
			total_returns += reward
		returns_list.append(total_returns)

		sum_of_10000_returns += total_returns

		# if (i + 1) % 10000 == 0:
		# 	wins, losses, ties = evaluate_greedy_policy(qlearning, env, testing_iter, display=False)
		# 	print('Iteration:', str(i + 1) + '/' + str(num_episodes))
		# 	print("Wins: ", wins)
		# 	wins_list.append(wins)
		# 	print("Losses: ", losses)
		# 	losses_list.append(losses)
		# 	print('Ties:', ties)
		# 	ties_list.append(ties)
		#
		#
		# 	average_of_10000_returns = sum_of_10000_returns / 10000
		# 	returns_list.append(average_of_10000_returns)
		# 	print('Iteration:', str(i + 1) + '/' + str(num_episodes))
		# 	print('Average Returns:', average_of_10000_returns)
		# 	sum_of_10000_returns = 0



		# # Code to save Q_Max at states
		# if i in episodes_to_save_Q:
		# 	max_Q = np.around(np.amax(qlearning.Q,axis=1)).T
		#
		#
		# 	print(max_Q[max_Q > 0])
		# 	# plt.savefig('{}_Iterations.png'.format(i))
		# 	# plt.show()
		# 	# plt.close()



		if i % qfunction_checkpoint == 0:

			#Plotting the Q-Function for the first move
			first_move_q = np.asarray([qlearning.Q[0][i] for i in range(9)]).reshape((3,3)).round(decimals=2)
			for (m, l), label in np.ndenumerate(first_move_q):
				plt.text(l, m, label, ha='center', va='center')
			plt.imshow(first_move_q)
			plt.savefig('results/First_Move/First_Move_Q_at_{}_Episodes'.format(i))

			if display_graphs:
				plt.show()
				plt.close()

			#Plotting the Q function for the second move when the first person goes in the top left
			second_moves_list = [env.qlearning.Q[1][i] for i in range(1,9)]
			second_moves_list.insert(0, 0)
			second_moves = np.asarray(second_moves_list).reshape((3,3)).round(decimals=2)
			for (m, l), label in np.ndenumerate(second_moves):
				plt.text(l, m, label, ha='center', va='center')
			plt.imshow(second_moves)
			plt.savefig('results/Second_Move/Move_After_Top_Left_at_{}_Episodes'.format(i))

			if display_graphs:
				plt.show()
				plt.close()


		if i % gameplay_checkpoint == 0:
			# evaluate the greedy policy to see how well it performs
			wins, losses, ties = evaluate_greedy_policy(qlearning, env, testing_iter,display=display_games)
			print("Wins: ", wins)
			wins_list.append(wins)
			print("Losses: ", losses)
			losses_list.append(losses)
			print('Ties:', ties, '\n')
			ties_list.append(ties)

	if save_bots:
		np.save("play_first_bot1", qlearning.Q)
		np.save("play_second_bot1", env.qlearning.Q)

	#This code is for plotting returns of the training policy on map 3
	plt.title('Q-Learning Test Runs')
	plt.ylabel('Percentage')
	plt.xlabel('Iterations in {}\'s'.format(gameplay_checkpoint))
	plt.plot(range(len(wins_list)), wins_list, label='Wins')
	plt.plot(range(len(losses_list)), losses_list, label='Losses')
	plt.plot(range(len(ties_list)), ties_list, label='Ties')
	plt.legend(loc='best')
	plt.savefig('results/Naiive_QLearning_Tic_Tac_Toe_Self_Play_X_Wins.png')

	if display_graphs:
		plt.show()
		plt.close()

	# #This code is for plotting returns of the training policy on map 3
	# plt.title('Q-Learning Training Runs')
	# plt.ylabel('Returns')
	# plt.xlabel('Iterations')
	# plt.plot(range(len(returns_list)), returns_list)
	# # plt.savefig('QLearning_Training_Map3.png')
	# plt.show()


	# #This code is for plotting the q-value of the first state on map 2
	# env.reset()
	# state = discretized_state(env)
	# optimal_first_state_Q = np.max(qlearning.Q[state])
	# plt.plot(range(num_episodes), first_state_Qs, label='Q-value Training')
	# plt.plot(range(num_episodes), [optimal_first_state_Q] * num_episodes, label='Optimal Q-value')
	# plt.title('Starting State Q-Value on Map2')
	# plt.ylabel('Q-Value of First State')
	# plt.xlabel('Iterations')
	# plt.legend(loc='best')
	# # plt.savefig('Starting_State_QValue_Map2.png')
	# plt.show()
	

if __name__ == '__main__':
	main()