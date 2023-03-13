

import numpy as np

# global variables
BOARD_ROWS = 5
BOARD_COLS = 5
WIN_STATE = (4, 4)
JUMP_STATE = (3, 3)
START = (1, 0)
JUMP_POSITION = (1, 3)
OBSTACLES_STATE = [(2, 2), (2, 3), (2, 4), (3, 2)]
CUMULATIVE_REWARD = 0
EPISODE_COUNT = 0
DETERMINISTIC = True


class State:
    def __init__(self, state=START):
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
        self.board[2, 2] = -1
        self.board[2, 3] = -1
        self.board[2, 4] = -1
        self.board[3, 2] = -1
        self.state = state
        self.isEnd = False
        self.determine = DETERMINISTIC
        # initial reward state
        self.episode_reward = 0
        self.state_values = {}

    def giveReward(self):
        if self.state == WIN_STATE:
            return 10
        elif self.state == JUMP_POSITION and self.state == JUMP_STATE:
            print("HERE IS +5 FOR YOU!")
            return 5
        else:
            return 0

    def isEndFunc(self):
        if (self.state == WIN_STATE) or (EPISODE_COUNT >= 30 and CUMULATIVE_REWARD > 10):
            self.isEnd = True

    def nxtPosition(self, action):
        """
        action: North, South, East, West, Jump
        -------------
        0 | 1 | 2| 3| 4|
        1 |
        2 |
        3 |
        4 |
        return next position
        """
        if self.determine:
            if action == "North":
                nxtState = (self.state[0] - 1, self.state[1])
            elif action == "South":
                nxtState = (self.state[0] + 1, self.state[1])
            elif action == "East":
                nxtState = (self.state[0], self.state[1] - 1)
            elif action == "Jump":
                nxtState = JUMP_STATE
            else:
                nxtState = (self.state[0], self.state[1] + 1)
            # if next state legal
            if (nxtState[0] >= 0) and (nxtState[0] <= (BOARD_ROWS - 1)):
                if (nxtState[1] >= 0) and (nxtState[1] <= (BOARD_COLS - 1)):
                    if nxtState != OBSTACLES_STATE:
                        return nxtState
            return self.state

    def showBoard(self):
        self.board[self.state] = 1
        for i in range(0, BOARD_ROWS):
            print('-----------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 1:
                    token = '*'
                if self.board[i, j] == -1:
                    token = 'z'
                if self.board[i, j] == 0:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-----------------')


# Agent of player
class Agent:

    def __init__(self):
        self.states = []
        self.actions = ["North", "South", "East", "West"]
        self.State = State()
        self.lr = 0.2
        self.exp_rate = 0.3

        # initial state reward
        self.state_values = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.state_values[(i, j)] = 0  # set initial value to 0

    def complete_episode(self):
        global EPISODE_COUNT
        global CUMULATIVE_REWARD
        new = EPISODE_COUNT + 1
        if EPISODE_COUNT >= 30:
            EPISODE_COUNT = 0
        else:
            new
        CUMULATIVE_REWARD += self.episode_reward
        self.episode_reward = 0

    def chooseAction(self):
        # choose action with most expected value
        mx_nxt_reward = 0
        action = ""
        available_actions = self.actions
        # providing the agent a jump option during action
        if self.State == JUMP_POSITION:
            available_actions = self.actions + ["jump"]
        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            for a in self.actions:
                # if the action is deterministic
                nxt_reward = self.state_values[self.State.nxtPosition(a)]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
        return action

    def takeAction(self, action):
        position = self.State.nxtPosition(action)
        return State(state=position)

    def reset(self):
        self.states = []
        self.State = State()

    def play(self, rounds=50):
        i = 0
        while i < rounds:
            # to the end of game back propagate reward
            if self.State.isEnd:
                # back propagate
                reward = self.State.giveReward()
                # explicitly assign end state to reward values
                self.state_values[self.State.state] = reward  # this is optional
                # An episode has been completed here
                self.episode_reward = reward
                self.complete_episode()
                print("Game End Reward", reward)
                for s in reversed(self.states):
                    reward = self.state_values[s] + self.lr * (reward - self.state_values[s])
                    self.state_values[s] = round(reward, 3)
                self.reset()
                i += 1
            else:
                action = self.chooseAction()
                # append trace
                self.states.append(self.State.nxtPosition(action))
                print("current position {} action {}".format(self.State.state, action))
                # by taking the action, it reaches the next state
                self.State = self.takeAction(action)
                # mark is end
                self.State.isEndFunc()
                print("nxt state", self.State.state)
                print("---------------------")

    def showValues(self):
        for i in range(0, BOARD_ROWS):
            print('----------------------------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                out += str(self.state_values[(i, j)]).ljust(6) + ' | '
            print(out)
        print('----------------------------------')


if __name__ == "__main__":
    ag = Agent()
    ag.play(100)
    print(ag.showValues())