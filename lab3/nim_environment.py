import numpy as np
from nim_utils import Nimply

#ACTIONS = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}

class NimBoard(object):
    def __init__(self, num_rows: int, k: int = None) -> None:
        self._rows = [i*2 + 1 for i in range(num_rows)]
        self._k = k
        self._winner = None
        
        self.steps = 0
        self.construct_allowed_states()

    def __bool__(self):
        return sum(self._rows) > 0

    def __str__(self):
        return "<" + " ".join(str(_) for _ in self._rows) + ">"

    @property
    def rows(self) -> tuple:
        return tuple(self._rows)

    @property
    def k(self) -> int:
        return self._k

    def nimming(self, ply: Nimply) -> None:
        row, num_objects = ply
        assert self._rows[row] >= num_objects
        assert self._k is None or num_objects <= self._k
        self._rows[row] -= num_objects

    def display_board(self) -> None:
        for i in range(len(self._rows)):
            print(f"row[{i+1}]:  \t" + "| " * self._rows[i])

    def set_winner(self, winner: int) -> None:
        self._winner = winner

    @property
    def winner(self) -> int:
        return self._winner

    def construct_allowed_states(self):
        self.allowed_states = create_possible_tuples(self.rows)    

    def update_board(self, action: Nimply):
        self.nimming(action)
        self.steps += 1 # add steps

    def is_game_over(self):
        # check if robot in the final position
        return sum(self.rows) == 0

    def get_state_and_reward(self, winner=None):
        return self.rows, self.give_reward(winner)

    def give_reward(self, winner):
        # if at end reward based on the fact if agent won or lost
        if winner is not None:
            return 2 if winner else -2 
        # if not at end give -1 reward
        return -1


def create_possible_tuples(rows):
        allowed = set()
        for i in range(rows[0]+1):
            lst = [i]
            add_el_to_tuple(rows, lst, allowed, row_idx=1)
        return allowed

def add_el_to_tuple(rows, lst, allowed, row_idx):
    if row_idx == len(rows):
        tup = tuple(sorted(lst))
        allowed.add(tup)
        return
    
    for elem in range(rows[row_idx]+1):
        lst.append(elem)
        add_el_to_tuple(rows, lst, allowed, row_idx+1)
        lst.pop()    
    return
