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

    def decide(self, S):
        """Choose action according to the soft policy"""
        if random() <= self.epsilon:
            return choice(list(self.Q[S].keys()))
        else:
            return max(self.Q[S], key=self.Q[S].get)

    def __str__(self):
        return "{}-greedy policy with value function:\n{}".format(self.epsilon, self.Q)

class Random(EpsilonGreedy):
    """100% random policy"""
    def __init__(self, Q):
        super().__init__(Q, 1.0)

    def decide(self, S):
        return choice(list(self.Q[S].keys()))

if __name__ == "__main__":
    Q = {0: {"left": 1.0, "right": 0.0}, 1: {"left": 0.0, "right": 1.0}}
    p = EpsilonGreedy(Q, epsilon=0.2)
    r = Random(Q)

    def mini_test(p, s, n=10):
        print("For state {}:".format(s), end=" ")
        for _ in range(10):
            print(p.decide(s), end=", ")
        print()

    print("soft:")
    mini_test(p, 0)
    mini_test(p, 1)
    print("random:")
    mini_test(r, 0)
    mini_test(r, 1)