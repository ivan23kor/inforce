import matplotlib.pyplot as plt
from env import RandomWalk, STANDARD_RANDOM_WALK
from policy import EpsilonGreedy, Random
# from random import choice # for episode initial state generation
from tqdm import tqdm

DEBUG = True

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

    def generate_episode(self, policy):
        """Generate episode from the given origin following the policy"""
        # Initial state
        S, A = 0, "empty_acton"
        # page 109 of the book: "all starting in the same state"
        # S, A = choice(self.env.states), "empty_acton"
        self.env.move_agent(S)

        episode = [] # --| Return of the function |-- [(A_0, R_1, S_0), ...,
                     #                     (terminal)  (A_T-1, R_T, S_T-1)]
        while not self.env.done():
            A = policy.decide(self.Q[S])
            S_prev = S
            R, S = self.env.move(A, move_agent=True)
            episode.append((S_prev, A, R))
        return episode

    def train(self, p, b, num_iter=100000, optimal_Q=None, disc_aware=False):
        self._init_train()

        if disc_aware:
            self.train_disc_aware(p, b, num_iter, optimal_Q)
        else:
            self.train_disc_unaware(p, b, num_iter, optimal_Q)

    def train_disc_unaware(self, p, b, num_iter, optimal_Q):
        history = []
        weight_A_S = len(self.env.actions)
        for _ in tqdm(range(num_iter)):
            G, W = 0.0, 1.0
            episode = self.generate_episode(b)
            for (S, A, R) in episode[::-1]:
                # G
                G = self.gamma * G + R
                # C
                self.C[S][A] += W
                # Q
                self.Q[S][A] += W / self.C[S][A] * (G - self.Q[S][A])

                ########### For epsilon-soft policies ###########
                # p_A_S = p.epsilon / len(self.Q[S])            #
                # if max(self.Q[S], key=self.Q[S].get) != A:    #
                #     p_A_S += 1 - p.epsilon                    #
                # W *= p_A_S / b_A_S                            #
                #################################################
                # However, for pure greedy policies we ensure that the
                # probability to choose A is nonzero it is nonzero for b
                if p.decide(self.Q[S]) != A:
                    break
                W *= weight_A_S

                if optimal_Q:
                    history.append(q_diff(optimal_Q, self.Q))

        return history

    def numerator(self, episode, p, b, t, last):
        """Computes numerator from the formula (5.10), p. 113
        indexes t and last are 1-based
        All indexing over the episode is shifted by -1 as it is represented as a
        Python array.
        Powers are not shifted as they are 1-based integers

        Observation for the greedy policy: rho(x:y) = weight_A_S ** (y - x + 1)
        """
        if t > last:
            return 0.0

        # rho matrix
        weight_A_S = len(self.env.actions)
        rho
        rho = [v for _ in range(t, last + 1)]

        # First additive
        first = 0.0
        for h in range(t + 1, last + 1):
            first += self.gamma**(h - t - 1)\
                   * rho(t:h - 1)\
                   * sum(el[2] for el in episode[t - 1:h - 1])

        # Second additive
        second = self.gamma**(last - t)\
               * rho(t:last)\
               * sum(el[2] for el in episode[t - 1:last])

        return (1 - self.gamma) * first + second

    def train_disc_aware(self, p, b, num_iter, optimal_Q):
        if DEBUG:
            # Just for now
            print("[Info] setting num_iter to 1 for debugging")
            num_iter = 1

        for _ in tqdm(range(num_iter)):
            # Per-episode inits
            new_Q = {state: {"numerator": 0.0, "denominator": 0.0}
                     for state in self.env.states}
            new_counts = {s: 0 for s in self.env.states}
            episode = self.generate_episode(b)
            last = len(episode)
            for (S, A, R) in episode:
                print("{} {} + {} yields {}".format(n, S, A, R))
                if p.decide(self.Q[S]) != A:
                    break
                new_counts[S] += 1
                new_Q[S]["numerator"] += self.numerator(p, b, new_counts[S], last)
        # print(new_Q)

def q_diff(q1, q2):
    ans = 0.0
    for state in q1:
        for action in q1[state]:
            ans += (q1[state][action] - q2[state][action])**2
    return ans**(0.5)

def print_Q(Q, shape):
    for row in range(shape[0]):
        for col in range(shape[1]):
            best_action = max(Q[row * shape[1] + col].items(), key=lambda kv: kv[1])
            print("\t{}|{:.1f}".format(best_action[0][0], best_action[1]), end="")
        print("\n\n")

def eval_unaware(algorithm, p, b, num_iter):
    # First train
    print("First train:")
    algorithm.train(p, b, num_iter)
    optimal_Q = algorithm.Q

    # Second train
    print("Second train:")
    history = algorithm.train(p, b, num_iter, optimal_Q)

    print("Best actions of Q:")
    print_Q(algorithm.Q, env.shape)

    print("Training history:")
    plt.plot(history)
    plt.show()

if __name__ == "__main__":
    env = RandomWalk(**STANDARD_RANDOM_WALK)
    algorithm = ROMC(env, 0.9)
    num_iter = 10000

    p = EpsilonGreedy(0.0) # Target policy
    b = Random() # Behavior policy

    # eval_unaware(algorithm, p, b, num_iter)
    algorithm.train(p, b, 1, None, True)