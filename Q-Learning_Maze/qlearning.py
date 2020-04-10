import numpy as np
from grid_world import *
import operator
import matplotlib.pyplot as plt

#Hyperparameters
#Learning rate
alpha = .5
#Number of episodes to run q-learning
num_episodes = 500
#Noise in training policy
eps = 0.01
#How many episodes to evaluate the policy after training
testing_iter = 1000
#Choosing a map environment from selection of maps in 'tictactoe_env.py'
Map = MAP4
#Number of times that full Q-Learning is run (results across runs are averaged)
num_runs = 20

#Function which gives the best policy given a Q function
def optimal_policy(Q, state):
    Q_next = {}
    for next_action in range(Q.shape[1]):
        Q_next[next_action] = Q[state, next_action]
    next_action, next_q = max(Q_next.items(), key=operator.itemgetter(1))
    return next_action, next_q

#The Q-Learning object which is responsible for holding the Q function and applying the Bellman Operator
class QLearning(object):
    # Initialize a Qlearning object
    # alpha is the "learning_rate"
    def __init__(self, num_states, num_actions, alpha=0.5):
         # initialize Q values to something
        self.Q = np.zeros((num_states, num_actions))
        self.alpha = alpha
        self.num_actions = num_actions


    #Applying the Bellman Operator
    def update(self, state, action, reward, next_state, done):
        if not done:
            max_next_action, max_next_Q = optimal_policy(self.Q, next_state)
        else:
            max_next_action, max_next_Q = None , 0

        self.Q[state, action] = ((1-self.alpha) * self.Q[state, action]) + self.alpha * (reward + max_next_Q)

        #return the next action also in case it's needed for time complexity reasons
        return max_next_action




#Test the policy to see how well it worked
def evaluate_greedy_policy(qlearning, env, niter=100):
    num_succ = 0
    for i in range(niter):
        done = False
        state = env.reset()
        action, _ = optimal_policy(qlearning.Q, state)
        reward = 0

        while not done:
            # Step
            next_state, reward, done = env.step(action)

            # Q update
            action, _ = optimal_policy(qlearning.Q, next_state)

        if reward == 100:
            num_succ += 1
    return 100 * num_succ/niter


if __name__ == "__main__":

    env = GridWorld(Map)
    num_states = env.get_num_states()
    episodes_to_save_Q = [0, 10, 100, 250, 500]
    episodes_to_save_success_rate = list(range(0, num_episodes + 1, 10))

    success_rate_list = np.zeros((num_runs, len(episodes_to_save_success_rate)))
    first_state_Qs = np.zeros((len(episodes_to_save_Q), num_runs, num_states))

    for run in range(num_runs):
        print("Run:",run+1)

        qlearning = QLearning(num_states, env.get_num_actions(), alpha=alpha)


        for i in range(num_episodes + 1):

            done = False
            state = env.reset()

            max_next_action, _ = optimal_policy(qlearning.Q, state)
            while not done:

                #Epsilon greedy policy
                random_action = np.random.choice(env.get_num_actions())
                action = np.random.choice([max_next_action, random_action], p= [1-eps, eps])

                #Step
                next_state, reward, done = env.step(action)

                #Q update
                max_next_action = qlearning.update(state, action, reward, next_state, done)


                state = next_state

            #Recording max qfunction
            if i in episodes_to_save_Q:
                first_state_Qs[episodes_to_save_Q.index(i), run] = np.max(qlearning.Q, axis=1)

            #Recording the success rates
            if i in episodes_to_save_success_rate:
                # evaluate the greedy policy to see how well it performs
                frac = evaluate_greedy_policy(qlearning, env, testing_iter)
                success_rate_list[run][i//10] = frac

    #PLotting average qfunction at each episode in 'episodes_to_save_Q'
    for i in range(len(episodes_to_save_Q)):
        currentQ = np.mean(first_state_Qs[i], axis=0).reshape((env.n_rows, env.n_cols)).round(decimals=2)

        for (m, l), label in np.ndenumerate(currentQ):
            plt.text(l, m, label, ha='center', va='center')
        plt.imshow(currentQ)
        plt.savefig('results/QFunction/QFunction_After_{}_Episodes'.format(episodes_to_save_Q[i]))

        plt.show()
        plt.close()

    # Plotting average returns over all runs
    success_rate_list_average = np.mean(success_rate_list, axis=0).round(decimals=2)
    plt.plot(range(len(success_rate_list_average)), success_rate_list_average)
    plt.title('Q-Learning Training')
    plt.ylabel('% Success w/ Optimal Policy')
    plt.xlabel('Iterations in 10\'s')
    plt.savefig('results/QLearning_Maze_Training.png')

    plt.show()
    plt.close()