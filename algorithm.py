from policy import Greedy, Random
from random import choice
from tqdm import tqdm

class OffPolicyMC(object):
    """Off-policy Monte Carlo conrol"""
    def __init__(self, actions, states, env, gamma=0.9, policy_epsilon=0.1):
        self.actions = actions
        self.states = states
        self.env = env
        self.gamma = gamma

        self.p = Greedy(actions, states, epsilon=policy_epsilon) # target policy

    def generate_episode(self, policy, S=[0, 0]):
        if not S:
            S = choice(self.states)
        episode = [] # Return of the function: [(A_0, R_1, S_1), ...,
                     #              (terminal)  (A_T-1, R_T, S_T)]
        A = "empty_acton"
        while S != self.env.goal:
            A = policy.decide()
            R, S = self.env.act(S, A)
            episode.append((A, R, tuple(S)))
        return episode

    def train(self):
        initial_prob = 1.0 / len(self.actions)
        b = Random(self.actions, self.states) # behavior policy
        C = {tuple(s): {a: 0 for a in self.actions} for s in self.states}
        for _ in range(1000):
        # while True:
            G, W = 0, 1
            episode = self.generate_episode(b)
            for (A, R, S) in episode[::-1]:
                # G
                G = self.gamma * G + R
                # C
                C[S][A] += W
                # Q
                self.p.Q[S][A] += W / C[S][A] * (G - self.p.Q[S][A])
                # # Break if policy changed
                # if self.p.decide(S)[1] != A:
                #     break
                W /= b.Q[S][A]

        print(self.p.Q)

class OnPolicyMC(object):
    """Off-policy Monte Carlo conrol"""
    def __init__(self, actions, states, env, gamma=0.9, policy_epsilon=0.1):
        self.actions = actions
        self.states = states
        self.env = env
        self.gamma = gamma

        self.p = Greedy(actions, states, epsilon=policy_epsilon) # target policy

    def generate_episode(self, policy, S=[0, 0]):
        if not S:
            S = choice(self.states)
        episode = [] # Return of the function: [(A_0, R_1, S_1), ...,
                     #              (terminal)  (A_T-1, R_T, S_T)]
        A = "empty_acton"
        while S != self.env.goal:
            A = policy.decide()
            R, S = self.env.act(S, A)
            episode.append((A, R, tuple(S)))
        return episode

    def train(self, num_iter=100000):
        initial_prob = 1.0 / len(self.actions)
        b = Random(self.actions, self.states) # behavior policy
        C = {tuple(s): {a: 0 for a in self.actions} for s in self.states}
        for _ in range(num_iter):
        # while True:
            G, W = 0, 1
            episode = self.generate_episode(b)
            for (A, R, S) in episode[::-1]:
                # G
                G = self.gamma * G + R
                # C
                C[S][A] += W
                # Q
                self.p.Q[S][A] += W / C[S][A] * (G - self.p.Q[S][A])
                # # Break if policy changed
                # if self.p.decide(S)[1] != A:
                #     break
                W /= b.Q[S][A]
