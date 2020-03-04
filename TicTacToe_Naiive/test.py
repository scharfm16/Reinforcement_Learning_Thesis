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

	wins_list = np.zeros((num_runs, num_episodes//gameplay_checkpoint+1))
	losses_list = np.zeros((num_runs, num_episodes//gameplay_checkpoint+1))
	ties_list = np.zeros((num_runs, num_episodes//gameplay_checkpoint+1))
	first_moves_across_runs = np.zeros((num_episodes//qfunction_checkpoint + 1, num_runs,  9))
	second_moves_across_runs = np.zeros((num_episodes//qfunction_checkpoint + 1, num_runs, 9))

	for run in range(num_runs):
		print("Run:",run+1)

		env = TicTacToe()
		qlearning = QLearning(num_states, num_actions, alpha=agent_alpha)

		if load_bots:
			qlearning.Q = np.load("play_first_bot1.npy")
			env.qlearning.Q = np.load("play_second_bot1.npy")

		for i in range(num_episodes+1):
			done = False
			env.reset()
			state = discretized_state(env)


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

			if i % qfunction_checkpoint == 0:

				#Plotting the Q-Function for the first move
				first_move_q = [qlearning.Q[0][i] for i in range(9)]
				first_moves_across_runs[i//qfunction_checkpoint][run] =np.asarray(first_move_q)

				#Plotting the Q function for the second move when the first person goes in the top left
				second_move_q = [0]+[env.qlearning.Q[1][i] for i in range(1,9)]
				second_moves_across_runs[i//qfunction_checkpoint][run] = np.asarray(second_move_q)


			if i % gameplay_checkpoint == 0:
				# evaluate the greedy policy to see how well it performs
				wins, losses, ties = evaluate_greedy_policy(qlearning, env, testing_iter,display=display_games)
				print("Wins: ", wins)
				wins_list[run][i//gameplay_checkpoint] = wins
				print("Losses: ", losses)
				losses_list[run][i//gameplay_checkpoint] = losses
				print('Ties:', ties, '\n')
				ties_list[run][i//gameplay_checkpoint] = ties

		if save_bots:
			np.save("play_first_bot1", qlearning.Q)
			np.save("play_second_bot1", env.qlearning.Q)


	print("Analyzing Results Over {} Runs:".format(num_runs))
	for i in range(num_episodes//qfunction_checkpoint + 1):

		first_moves_average = np.mean(first_moves_across_runs[i], axis=0).reshape((3,3)).round(decimals=2)
		for (m, l), label in np.ndenumerate(first_moves_average):
			plt.text(l, m, label, ha='center', va='center')
		plt.imshow(first_moves_average)
		plt.savefig('results/First_Move/Average_First_Move_Q_at_{}_Episodes'.format(i * qfunction_checkpoint))

		if display_graphs:
			plt.show()
			plt.close()

		second_moves_average = np.mean(second_moves_across_runs[i], axis=0).reshape((3,3)).round(decimals=2)
		for (m, l), label in np.ndenumerate(second_moves_average):
			plt.text(l, m, label, ha='center', va='center')
		plt.imshow(second_moves_average)
		plt.savefig('results/Second_Move/Average_After_Top_Left_at_{}_Episodes'.format(i * qfunction_checkpoint))

		if display_graphs:
			plt.show()
			plt.close()

	average_wins = np.mean(wins_list, axis=0)
	average_losses = np.mean(losses_list, axis=0)
	average_ties = np.mean(ties_list, axis=0)

	# This code is for plotting returns of the training policy on map 3
	plt.title('Naiive Q-Learning Test Runs')
	plt.ylabel('Percentage')
	plt.xlabel('Iterations in {}\'s'.format(gameplay_checkpoint))
	plt.plot(range(len(average_wins)), average_wins, label='Wins')
	plt.plot(range(len(average_losses)), average_losses, label='Losses')
	plt.plot(range(len(average_ties)), average_ties, label='Ties')
	plt.legend(loc='best')
	plt.savefig('results/Naiive_QLearning_Tic_Tac_Toe_Average_X_Wins.png')

	if display_graphs:
		plt.show()
		plt.close()
	

if __name__ == '__main__':
	main()