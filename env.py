# =============================================================================
#
#                                COPYRIGHT THING
#
#      IVAN KOROSTELEV DID THIS IN 2020, DO NOT COPY WITHOUT THIS HEADER
#
# =============================================================================

from collections import OrderedDict
import numpy as np

ACTIONS = OrderedDict({"left": [0, -1], "up": [-1, 0], "right": [0, 1],
                       "down": [1, 0]})

# Rewards
R = {
    "BORDER": -1, # Reward for hitting the border: do not move the agent and
                  # give it this reward instead
    "CELL": -1, # A cell that is not the goal or a pit
    "GOAL": 10,
    "MOVE": -1, # Each actions costs something
    "PIT": -10, # Covered with dirt, unpleasant to be in, but does not restart
                # the agent to the origin
}

class Env:
    """Environment provides lists of states and actions,
    rewards and the gamma factor. Probability of transition is not implemented
    so every action deterministically leads to a state.
    """

    def __init__(self, shape, goal, pits, actions=ACTIONS, R=R):
        """
        Parameters
        ----------
        shape : tuple, optional
            size of the random walk board
        goal : list, optional
            position of the goal
        pits : list of lists
            positions of the pits
        actions : list, optional
            allowed actions and their coordinates
        R : dict, optional
            rewards per each possible cell/action
        """
        # Make board
        self.shape = shape
        self.states = [(x, y) for x in range(shape[0]) for y in range(shape[1])]
        if not isinstance(goal, list):
            raise ValueError("Agrument goal should be a list, e.g.: [10, 10]")
        self.goal = goal
        if not isinstance(pits, list) or not all((isinstance(el, list) for el in pits)):
            raise ValueError("Agrument pits should be a list of lists, e.g.: [[0, 2], [3, 4]]")
        self.pits = pits
        self.actions = actions
        self.R = R # Rewards

    def __str__(self):
        cell_str = {"CELL": ".", "GOAL": "o", "PIT": "x"}
        ans = [" ".join(cell_str[cell] for cell in row) for row in self.cells]
        return "\n\n".join(ans)

    def act(self, S, A):
        R = self.R["MOVE"]
        S_prime = [S[0] + self.actions[A][0], S[1] + self.actions[A][1]]

        if S_prime[0] < 0 or S_prime[0] >= self.shape[0] or S_prime[1] < 0\
                            or S_prime[1] >= self.shape[1]:
            R = self.R["BORDER"]
            S_prime = S
        elif S_prime == self.goal:
            R = self.R["GOAL"]
        elif S_prime in self.pits:
            R = self.R["PIT"]

        return R, S_prime
