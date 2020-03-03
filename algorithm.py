import matplotlib.pyplot as plt
from copy import copy, deepcopy
from env import RandomWalk, STANDARD_RANDOM_WALK
from policy import EpsilonGreedy, Random
from random import choice
from tqdm import tqdm

def q_diff(q1, q2):
    ans = 0.0
    for s in q1:
        for a in q1[s]:
            ans += abs(q1[s][a] - q2[s][a])
    return ans

class ROMC(object):
    """Random behavior policy Off-policy Monte Carlo conrol"""
    def __init__(self, env, gamma=0.1, policy_epsilon=0.1):
        self.env = env
        self.gamma = gamma

        # Training variables
        self._init_train()

    def _init_train(self):
        init_prob = 1.0 / len(self.env.actions)
        init_prob = 0.0
        self.Q = {s: {a: init_prob for a in self.env.actions}
                                   for s in self.env.states}
        self.C = {s: {a: 0.0 for a in self.env.actions}
                             for s in self.env.states}
        self.G_history = []

    def generate_episode(self, policy):
        """Generate episode from the given origin following the policy"""
        # Initial state
        S, A = choice(self.env.states), "empty_acton"
        self.env.move_agent(S)

        episode = [] # --| Return of the function |-- [(A_0, R_1, S_0), ...,
                     #                     (terminal)  (A_T-1, R_T, S_T-1)]
        while not self.env.done():
            A = policy.decide(S)
            S_prev = S
            R, S = self.env.move(A, move_agent=True)
            episode.append((A, R, S_prev))
        return episode

    def train(self, p, b, num_iter=100000, disc_aware=False):
        self._init_train()
        self.q_diff = []

        for _ in tqdm(range(num_iter)):
            G, W = 0.0, 1.0
            episode = self.generate_episode(b)
            initial_Q = deepcopy(p.Q)
            for (A, R, S) in episode[::-1]:
                # G
                G = self.gamma * G + R
                self.G_history.append(G)
                # C
                self.C[S][A] += W
                # Q
                if not disc_aware:
                    p.Q[S][A] += W / self.C[S][A] * (G - p.Q[S][A])
                else:
                    p.Q[S][A] += W / self.C[S][A] * (G - p.Q[S][A])
                # Break if policy changed
                if p.decide(S) != A:
                    break
                W *= len(self.env.actions)
            self.q_diff.append(q_diff(initial_Q, p.Q))

def print_Q(Q, shape):
    for row in range(shape[0]):
        for col in range(shape[1]):
            best_action = max(Q[row * shape[1] + col].items(), key=lambda kv: kv[1])
            print("\t{}|{:.1f}".format(best_action[0][0], best_action[1]), end="")
        print("\n\n")

if __name__ == "__main__":
    env = RandomWalk(**STANDARD_RANDOM_WALK)
    algorithm = ROMC(env, 0.9)

    p = EpsilonGreedy(algorithm.Q, 0.0) # Target policy
    b = Random(algorithm.Q) # Behavior policy
    algorithm.train(p, b, num_iter=10000, disc_aware=False)
    #algorithm.train(p, b, num_iter=10000, disc_aware=True)
    print_Q(p.Q, env.shape)
    # plt.plot(algorithm.q_diff)
    # plt.show()
