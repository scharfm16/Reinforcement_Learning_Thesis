from tictactoe_env import TicTacToe
import matplotlib.pyplot as plt
from Minimax_Q_Learning import *
import itertools
#HYPERPARAMETERS IN Q-LEARNING. RUN CODE IN TEST.PY

# run the greedy policy (no randomness) on the environment for niter number of times
# and return the fraction of times that it reached the goal
def evaluate_greedy_policy(qlearning, niter=100, display=True):
    env = TicTacToe()
    num_wins = 0
    num_losses = 0
    num_ties = 0
    for i in range(niter):
        reward = 0
        done = False
        env.reset()
        state = discretized_state(env)

        if display:
            print(env.agent_state)

        while not done:
            min_max_next_action, _ = optimal_policy(qlearning.Q, state, env.agent_state)

            if min_max_next_action[1] == None:
                action = min_max_next_action
            else:
                legal_action_mask = legal_actions(env.agent_state)
                actions = list(zip(*np.nonzero(legal_action_mask)))
                rand_action_idx = np.random.choice(len(actions))
                random_action = actions[rand_action_idx]
                chosen_action_index = np.random.choice([0,1], p=[1-testing_eps,testing_eps])

                action = [min_max_next_action, random_action][chosen_action_index]

            #Step
            dummy, reward, done = env.step(action)
            if (np.sum(env.agent_state == 1) != np.sum(env.agent_state == 2)) and (done==False):
                print('OH JEEZ OH BOI THIS IS A VERY BAD ERROR FIGURE OUT WHATS GOING ON PLS')
            if display:
                print(env.agent_state)
            state = discretized_state(env)

        if reward == 10:
            num_wins += 1
        elif reward == -10:
            num_losses += 1
        elif reward == 0:
            num_ties += 1
    return (100* num_wins/niter, 100* num_losses/niter, 100*num_ties/niter)

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



def main():
    num_states = 3 ** 9
    num_actions = 9
    returns_list = []
    wins_list = []
    losses_list = []
    ties_list = []
    sum_of_10000_returns = 0
    env = TicTacToe()
    qlearning = Minimax_QLearning(num_states, num_actions, alpha=alpha)

    if load_bots:
        qlearning.Q = np.load("saved_Q.npy")

    for i in range(1, num_episodes+1):
        total_returns = 0
        done = False
        env.reset()
        state = discretized_state(env)

        min_max_next_action, _ = optimal_policy(qlearning.Q, state, env.agent_state)

        while not done:
            # print('state:', env.agent_state)
        #Epsilon greedy policy
            if min_max_next_action[1] == None:
                action = min_max_next_action
            else:
                legal_action_mask = legal_actions(env.agent_state)
                actions = list(zip(*np.nonzero(legal_action_mask)))
                rand_action_idx = np.random.choice(len(actions))
                random_action = actions[rand_action_idx]
                chosen_action_index = np.random.choice([0,1], p=[1-eps,eps])
                action = [min_max_next_action, random_action][chosen_action_index]
            # print("action:", action)
            #Step
            dummy, reward, done = env.step(action)

            if (np.sum(env.agent_state == 1) != np.sum(env.agent_state == 2)) and (done==False):
                print('OH JEEZ OH BOI THIS IS A VERY BAD ERROR FIGURE OUT WHATS GOING ON PLS')

            # print("reward", reward)
            # print("done:",done)
            next_state = discretized_state(env)
            # print("next state:",env.agent_state)
            #Q update
            min_max_next_action = qlearning.update(state, action, reward, next_state, done, env.agent_state)

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

        if i % 1000 == 0:
            print(i)

        if i % qfunction_checkpoint == 0:

            #Plotting the Q-Function for the first move
            first_move_q = np.asarray([np.nanmin(qlearning.Q[0][i][~i]) for i in range(9)]).reshape((3,3)).round(decimals=2)
            for (m, l), label in np.ndenumerate(first_move_q):
                plt.text(l, m, label, ha='center', va='center')
            plt.imshow(first_move_q)
            plt.savefig('First_Move_Q_at_{}_Episodes'.format(i))
            plt.show()
            plt.close()

            #Plotting the Q function for the second move when the first person goes in the top left
            second_moves = -np.asarray([0]+[qlearning.Q[0][0][i] for i in range(1,9)]).reshape((3,3)).round(decimals=2)
            for (m, l), label in np.ndenumerate(second_moves):
                plt.text(l, m, label, ha='center', va='center')
            plt.imshow(second_moves)
            plt.savefig('Move_After_Top_Left_at_{}_Episodes'.format(i))
            plt.show()
            plt.close()

        if i % gameplay_checkpoint == 0:
            # evaluate the greedy policy to see how well it performs
            wins, losses, ties = evaluate_greedy_policy(qlearning, testing_iter,display=display_games)
            print("Wins: ", wins)
            wins_list.append(wins)
            print("Losses: ", losses)
            losses_list.append(losses)
            print('Ties:', ties)
            ties_list.append(ties)

    if save_bots:
        np.save("saved_Q.npy", qlearning.Q)

    #This code is for plotting returns of the training policy on map 3
    plt.title('Minimax Q-Learning Test Runs')
    plt.ylabel('Percentage')
    plt.xlabel('Iterations in {}\'s'.format(gameplay_checkpoint))
    plt.plot(range(len(wins_list)), wins_list, label='Wins')
    plt.plot(range(len(losses_list)), losses_list, label='Losses')
    plt.plot(range(len(ties_list)), ties_list, label='Ties')
    plt.legend(loc='best')
    plt.savefig('Minimax_QLearning_Tic_Tac_Toe_Self_Play_X_Wins.png')
    plt.show()

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