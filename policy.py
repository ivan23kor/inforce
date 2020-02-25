from random import choice, random

class Greedy():
    """Epsilon-greedy (may be pure greedy) policy """

    def __init__(self, actions, states, epsilon=0):
        """
        Parameters
        ----------
        actions : dict
            allowed actions
        states : list
            list of possible states
        epsilon : float
            number within [0, 1] specifying percentage of time the
            action shouild be taken randomly
        """
        self.epsilon = epsilon
        self.actions = actions
        self.states = states

        initial_prob = 1.0 / len(self.actions)
        self.Q = {state: {a: initial_prob for a in self.actions}
                         for state in states}

    def decide(self, S):
        """Choose action according to the policy"""
        if random() <= self.epsilon:
            return choice(list(self.actions.keys()))
        else:
            return max(self.Q[S], key=self.Q[S].get)

class Random(Greedy):
    """100% random policy"""
    def __init__(self, *args):
        super().__init__(*args, 1.0)

    def decide(self):
        return choice(list(self.actions.keys()))
