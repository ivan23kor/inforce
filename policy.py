from random import choice, random

Q = {""}

class EpsilonGreedy():
    """Epsilon-greedy policy """

    def __init__(self, Q, epsilon=0):
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
        self.Q = Q

    def __str__(self):
        return "{}-greedy policy with value function:\n{}".format(self.epsilon, self.Q)

    def decide(self, S):
        """Choose action according to the soft policy"""
        if random() <= self.epsilon:
            return choice(list(self.Q[S].keys()))
        else:
            return max(self.Q[S], key=self.Q[S].get)

class Random(EpsilonGreedy):
    """100% random policy"""
    def __init__(self, *args):
        super().__init__(*args)

    def decide(self, S):
        return choice(list(self.Q[S].keys()))
