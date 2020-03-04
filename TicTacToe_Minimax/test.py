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

    wins_list = np.zeros((num_runs, num_episodes // gameplay_checkpoint + 1))
    losses_list = np.zeros((num_runs, num_episodes // gameplay_checkpoint + 1))
    ties_list = np.zeros((num_runs, num_episodes // gameplay_checkpoint + 1))
    first_moves_across_runs = np.zeros((num_episodes // qfunction_checkpoint + 1, num_runs, 9))
    second_moves_across_runs = np.zeros((num_episodes // qfunction_checkpoint + 1, num_runs, 9))

    for run in range(num_runs):
        print("Run:", run+1)
        env = TicTacToe()
        qlearning = Minimax_QLearning(num_states, num_actions, alpha=alpha)

        if load_bots:
            qlearning.Q = np.load("saved_Q.npy")

        for i in range(num_episodes+1):

            done = False
            env.reset()
            state = discretized_state(env)

            min_max_next_action, _ = optimal_policy(qlearning.Q, state, env.agent_state)

            while not done:
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

                next_state = discretized_state(env)

                #Q update
                min_max_next_action = qlearning.update(state, action, reward, next_state, done, env.agent_state)

                state = next_state



            if i % qfunction_checkpoint == 0:

                #Plotting the Q-Function for the first move
                first_move_q = [np.nanmin(qlearning.Q[0][i][~i]) for i in range(9)]
                first_moves_across_runs[i // qfunction_checkpoint][run] = np.asarray(first_move_q)

                # Plotting the Q function for the second move when the first person goes in the top left
                second_move_q = [0]+[qlearning.Q[0][0][i] for i in range(1,9)]
                second_moves_across_runs[i // qfunction_checkpoint][run] = -np.asarray(second_move_q)

            if i % gameplay_checkpoint == 0:
                # evaluate the greedy policy to see how well it performs
                wins, losses, ties = evaluate_greedy_policy(qlearning, testing_iter, display=display_games)
                print("Wins: ", wins)
                wins_list[run][i // gameplay_checkpoint] = wins
                print("Losses: ", losses)
                losses_list[run][i // gameplay_checkpoint] = losses
                print('Ties:', ties, '\n')
                ties_list[run][i // gameplay_checkpoint] = ties

        if save_bots:
            np.save("saved_Q.npy", qlearning.Q)

    print("Analyzing Results Over {} Runs:".format(num_runs))
    for i in range(num_episodes // qfunction_checkpoint + 1):

        first_moves_average = np.mean(first_moves_across_runs[i], axis=0).reshape((3, 3)).round(decimals=2)
        for (m, l), label in np.ndenumerate(first_moves_average):
            plt.text(l, m, label, ha='center', va='center')
        plt.imshow(first_moves_average)
        plt.savefig('results/First_Move/Average_First_Move_Q_at_{}_Episodes'.format(i * qfunction_checkpoint))

        if display_graphs:
            plt.show()
            plt.close()

        second_moves_average = np.mean(second_moves_across_runs[i], axis=0).reshape((3, 3)).round(decimals=2)
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
    plt.title('Minimax Q-Learning Test Runs')
    plt.ylabel('Percentage')
    plt.xlabel('Iterations in {}\'s'.format(gameplay_checkpoint))
    plt.plot(range(len(average_wins)), average_wins, label='Wins')
    plt.plot(range(len(average_losses)), average_losses, label='Losses')
    plt.plot(range(len(average_ties)), average_ties, label='Ties')
    plt.legend(loc='best')
    plt.savefig('results/Minimax_QLearning_Tic_Tac_Toe_Average_X_Wins.png')

    if display_graphs:
        plt.show()
        plt.close()





if __name__ == '__main__':
    main()