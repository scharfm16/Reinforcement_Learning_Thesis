import numpy as np 
from math import inf as infinity
import pdb
from Q_Learning import *

#HYPERPARAMETERS IN Q-LEARNING. RUN CODE IN TEST.

class TicTacToe(object):
    def __init__(self):
        self.game_state = [[' ',' ',' '],
                    [' ',' ',' '],
                    [' ',' ',' ']]

        self.agent_state = np.zeros((3,3)) #actual agent observation
        self.qlearning = QLearning(num_states=3 ** 9, num_actions = 9, alpha=env_alpha)
        self.last_state = None
        self.last_action = None
        self.eps = env_eps


    def print_board(self):
        print('----------------')
        print('| ' + str(self.game_state[0][0]) + ' || ' + str(self.game_state[0][1]) + ' || ' + str(self.game_state[0][2]) + ' |')
        print('----------------')
        print('| ' + str(self.game_state[1][0]) + ' || ' + str(self.game_state[1][1]) + ' || ' + str(self.game_state[1][2]) + ' |')
        print('----------------')
        print('| ' + str(self.game_state[2][0]) + ' || ' + str(self.game_state[2][1]) + ' || ' + str(self.game_state[2][2]) + ' |')
        print('----------------')

    def convert_state(self):
        #agent plays X. Env plays O. If cell is empty, denoted by zero
        #if it has X it is denoted by 1. if it has O it is denoted by 2.

        for i in range(3):
            for j in range(3):
                if self.game_state[i][j] == ' ':
                    self.agent_state[i][j] = 0
                elif self.game_state[i][j] == 'X':
                    self.agent_state[i][j] = 1
                else:
                    self.agent_state[i][j] = 2

        return self.agent_state

    def discretized_state(self):
        state = 0
        place = 0
        for i in range(3):
            for j in range(3):
                state += self.agent_state[i][j] * (3 ** place)
                place += 1
        return state

    def reset(self):
        self.game_state = [[' ',' ',' '],[' ',' ',' '],[' ',' ',' ']]
        current_state = "Not Done"

        # self.print_board()

        self.last_state = None
        self.last_action = None
        self.last_board = None

        winner = None
        current_player_idx = 0

        return self.convert_state()


    def step(self, action, default_strat=True, learning=True, display=False, bot_train=True):
        #action must be valid and must be numbered  0 through 8.
        #convert accordingly. Additionally write script in the
        #training/inference loop to make sure action is valid, i.e
        #do not output an action in a cell that is already occupied.
        #This will raise an exception and break your training loop.
        self.play_move('X', action)
        self.convert_state()
        if display:
            self.print_board()
        #Opponent plays action
        block_choice = self.getOpponentMove(self.game_state,"O", default_strat=default_strat, learning=learning, bot_train=bot_train)

        if block_choice != -10 and block_choice != 10 and block_choice != 0:
            self.play_move('O', block_choice)
        if display:
            self.print_board()
        rew, done = self.rew_calc()

        winner, current_state = self.check_current_state(self.game_state)
        if display:
            if current_state is "Draw":
                print("Draw")
            elif winner == 'X':
                print("Win")
            elif winner == 'O':
                print("Lost")

        return self.convert_state(), rew, done

    def rew_calc(self):
        reward = 0
        done = False
        current_state = "Not Done"

        winner, current_state = self.check_current_state(self.game_state)

        #While game is being played return done = False
        #Design the reward to be returned
        if current_state is "Not Done":
            reward = 0
            return reward, done

        if current_state is "Draw":
            reward = 0
            done = True
            return reward, done
        elif winner == 'X':
            reward = 10
            done = True
        elif winner == 'O':
            reward = -10
            done = True

        return reward, done



    def play_move(self, player, block_num):
        if self.game_state[int((block_num-1)/3)][(block_num-1)%3] is ' ':
            self.game_state[int((block_num-1)/3)][(block_num-1)%3] = player
        else:
            raise Exception('Invalid Action!')


    def play_move_hallucinate(self,state, player, block_num):
        if state[int((block_num-1)/3)][(block_num-1)%3] is ' ':
            state[int((block_num-1)/3)][(block_num-1)%3] = player
        else:
            raise Exception('Invalid Action!')



    def getOpponentMove(self, state,player, default_strat=True, learning=True, bot_train=True):
        winner_loser , done = self.check_current_state(state)
        if done == "Done" and winner_loser == 'O': # If Opponent won
            if not default_strat:
                self.qlearning.update(self.last_state, self.last_action, 10, None, True, self.agent_state, agent=False)
            return 10
        elif done == "Done" and winner_loser == 'X': # If Human won
            if not default_strat:
                self.qlearning.update(self.last_state, self.last_action, -10, None, True, self.agent_state, agent=False)
            return -10
        elif done == "Draw":    # Draw condition
            if not default_strat:
                self.qlearning.update(self.last_state, self.last_action, 0, None, True, self.agent_state, agent=False)
            return 0
        reward = 0

        if not default_strat:

            state = discretized_state(self)
            if self.last_state != None:
                self.qlearning.update(self.last_state, self.last_action, reward, state, False, self.agent_state, agent=False)

            max_next_action, _ = optimal_policy(self.qlearning.Q, state, self.agent_state, agent=False)
            legal_action_mask = legal_actions(self.agent_state)

            if learning and bot_train:
                #Epsilon greedy policy
                actions = [action for action in list(range(9)) if legal_action_mask[action]]
                random_action = np.random.choice(actions)
                action = np.random.choice([max_next_action, random_action], p=[1 - self.eps, self.eps])
            else:
                action = max_next_action



            self.last_state = state
            self.last_action = action
            self.last_board = self.agent_state.copy()
            return action + 1

        else:
            moves = []
            empty_cells = []
            for i in range(3):
                for j in range(3):
                    if state[i][j] is ' ':
                        empty_cells.append(i*3 + (j+1))

            for empty_cell in empty_cells:
                move = {}
                move['index'] = empty_cell
                new_state = self.copy_game_state(state) #hallucinate through states
                self.play_move_hallucinate(new_state, player, empty_cell)
                if player == 'O':    # If Opponent
                    result = self.getOpponentMove(new_state, 'X')    # make more depth tree for human
                    move['score'] = result
                else:
                    result = self.getOpponentMove(new_state, 'O')    # make more depth tree for Opponent
                    move['score'] = result

                moves.append(move)

            # Find best move
            best_move = None
            if player == 'O':   # If Opponent player
                best = -infinity
                for move in moves:
                    if move['score'] > best:
                        best = move['score']
                        best_move = move['index']
            else:
                best = infinity
                for move in moves:
                    if move['score'] < best:
                        best = move['score']
                        best_move = move['index']

            return best_move

    def copy_game_state(self,state):
        new_state = [[' ',' ',' '],[' ',' ',' '],[' ',' ',' ']]
        for i in range(3):
            for j in range(3):
                new_state[i][j] = state[i][j]
        return new_state

    def check_current_state(self,game_state):

        # Check horizontals
        if (game_state[0][0] == game_state[0][1] and game_state[0][1] == game_state[0][2] and game_state[0][0] is not ' '):
            return game_state[0][0], "Done"
        if (game_state[1][0] == game_state[1][1] and game_state[1][1] == game_state[1][2] and game_state[1][0] is not ' '):
            return game_state[1][0], "Done"
        if (game_state[2][0] == game_state[2][1] and game_state[2][1] == game_state[2][2] and game_state[2][0] is not ' '):
            return game_state[2][0], "Done"

        # Check verticals
        if (game_state[0][0] == game_state[1][0] and game_state[1][0] == game_state[2][0] and game_state[0][0] is not ' '):
            return game_state[0][0], "Done"
        if (game_state[0][1] == game_state[1][1] and game_state[1][1] == game_state[2][1] and game_state[0][1] is not ' '):
            return game_state[0][1], "Done"
        if (game_state[0][2] == game_state[1][2] and game_state[1][2] == game_state[2][2] and game_state[0][2] is not ' '):
            return game_state[0][2], "Done"

        # Check diagonals
        if (game_state[0][0] == game_state[1][1] and game_state[1][1] == game_state[2][2] and game_state[0][0] is not ' '):
            return game_state[1][1], "Done"
        if (game_state[2][0] == game_state[1][1] and game_state[1][1] == game_state[0][2] and game_state[2][0] is not ' '):
            return game_state[1][1], "Done"

        # Check if draw
        draw_flag = 0
        for i in range(3):
            for j in range(3):
                if game_state[i][j] is ' ':
                    draw_flag = 1
        if draw_flag is 0:
            return None, "Draw"


        return None, "Not Done"

if __name__ == '__main__':
	env = TicTacToe()
	env.getOpponentMove(env.agent_state,'O')