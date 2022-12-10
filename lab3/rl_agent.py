import numpy as np
from nim_utils import Nimply
from nim_environment import NimBoard


class Agent(object):
    def __init__(self, states: NimBoard, alpha=0.15, random_factor=0.2):  # 80% explore, 20% exploit
        self.actions =  [
        (r, o) for r, c in enumerate(states.rows) for o in range(1, c + 1) if states.k is None or o <= states.k
        ]
        
        self.state_history = [(states.rows, 0)]  # state: tuple(num_obj in rows), reward
        self.alpha = alpha
        self.random_factor = random_factor
        self.G = {}
        self.init_reward(states.allowed_states)

        self.won = 0
        self.played = 0

    def init_reward(self, allowed_states):
        for state in allowed_states:
            self.G[state] = np.random.uniform(low=1.0, high=0.1)

    def choose_action(self, rows):
        actions = [
        (r, o) for r, c in enumerate(rows) for o in range(1, c + 1)
        ]

        maxG = -10e15
        next_move = None
        randomN = np.random.random()
        if randomN < self.random_factor:
            # if random number below random factor, choose random action
            next_move_idx = np.random.choice(range(len(actions)))
            next_move = actions.pop(next_move_idx)
        else:
            # if exploiting, gather all possible actions and choose one with the highest G (reward)
            for action in actions:
                row_idx, obj_taken = action
                obj_left = rows[row_idx] - obj_taken
                new_state = rows[:row_idx] + (obj_left,) + rows[row_idx+1:]
                new_state = tuple(sorted(new_state))

                if self.G[new_state] >= maxG:
                    next_move = action
                    maxG = self.G[new_state]

        return Nimply(next_move[0], next_move[1])

    def update_state_history(self, rows, reward):
        self.state_history.append((rows, reward))

    def update_results(self, winner):
        if winner:
            self.won += 1
        self.played += 1

    def get_avg_wins(self):
        return self.won / self.played

    def reset_results(self):
        self.won = 0
        self.played = 0

    def learn(self):
        target = 0

        for prev, reward in reversed(self.state_history):
            prev = tuple(sorted(prev))
            self.G[prev] = self.G[prev] + self.alpha * (target - self.G[prev])
            target += reward

        self.state_history = []

        self.random_factor -= 10e-5  # decrease random factor each episode of play
