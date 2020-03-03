import numpy as np
from grid_world import *
import operator
import matplotlib.pyplot as plt

#Hyperparameters
alpha = .5
num_episodes = 1001
eps = 0.01
testing_iter = 1000
Map = MAP4



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
            action = qlearning.update(state, action, reward, next_state, done)

            state = next_state
        if reward >= 90:
            num_succ += 1
    return 100 * num_succ/niter


if __name__ == "__main__":

    env = GridWorld(Map)
    qlearning = QLearning(env.get_num_states(), env.get_num_actions(), alpha=alpha)
    returns_list = []
    first_state_Qs = []
    sum_of_100_returns = 0
    episodes_to_save_Q = [0, 10, 100, 500, 1000]

    for i in range(num_episodes):

        total_returns = 0
        done = False
        state = env.reset()

        first_state_Q = np.max(qlearning.Q[state])
        first_state_Qs.append(first_state_Q)

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
            total_returns += reward

        returns_list.append(total_returns)

        # # Code to save Q_Max at states on Map 4
        # if i in episodes_to_save_Q:
        #     flattened_max_Q = np.amax(qlearning.Q,axis=1)
        #     max_Q = np.around(flattened_max_Q.reshape((env.n_rows, env.n_cols)))

        #     for (m, l), label in np.ndenumerate(max_Q):
        #         plt.text(l, m, label, ha='center', va='center')
        #     plt.imshow(max_Q)
        #     # plt.savefig('{}_Iterations.png'.format(i))
        #     plt.show()
        #     plt.close()



    # print("Rolling out final policy")
    # done = False
    # state = env.reset()
    # action, _ = optimal_policy(qlearning.Q, state)
    #
    # while not done:
    #     next_state, reward, done = env.step(action)
    #     action = qlearning.update(state, action, reward, next_state, done)
    #     state = next_state
    #     env.print()

    # evaluate the greedy policy to see how well it performs
    frac = evaluate_greedy_policy(qlearning, env, testing_iter)
    print("Finding goal " + str(frac) + "% of the time.")

    #This code is for plotting returns of the training policy on map 3
    plt.plot(range(num_episodes), returns_list)
    plt.title('Q-Learning Training on Map4')
    plt.ylabel('Returns')
    plt.xlabel('Iterations')
    plt.savefig('QLearning_Training_Map4.png')


    # #This code is for plotting the q-value of the first state on map 2
    # state = env.reset()
    # optimal_first_state_Q = np.max(qlearning.Q[state])
    # plt.plot(range(num_episodes), first_state_Qs, label='Q-value Training')
    # plt.plot(range(num_episodes), [optimal_first_state_Q] * num_episodes, label='Optimal Q-value')
    # plt.title('Starting State Q-Value on Map2')
    # plt.ylabel('Q-Value of First State')
    # plt.xlabel('Iterations')
    # plt.legend(loc='best')
    # # plt.savefig('Starting_State_QValue_Map2.png')
    # plt.show()

