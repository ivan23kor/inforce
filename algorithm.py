import matplotlib.pyplot as plt
from env import RandomWalk, STANDARD_RANDOM_WALK
from policy import EpsilonGreedy, Random
# from random import choice # for episode initial state generation
from tqdm import tqdm

DEBUG = True
OPTIMAL_Q = None

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
            return self.train_disc_aware(p, b, num_iter, optimal_Q)
        else:
            return self.train_disc_unaware(p, b, num_iter, optimal_Q)

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
                history.append(mean_sqerror(optimal_Q, self.Q))

        return history

    def numerator(self, episode, t, last):
        """Computes numerator for the formula (5.10), p. 113
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
        rho = 1.0

        # First additive
        first = 0.0
        for h in range(t + 1, last + 1):
            rho *= weight_A_S
            first += self.gamma**(h - t - 1) * rho\
                   * sum(el[2] for el in episode[t - 1:h - 1])

        rho *= weight_A_S
        # Second additive
        second = self.gamma**(last - t) * rho\
               * sum(el[2] for el in episode[t - 1:last])

        return (1 - self.gamma) * first + second

    def denominator(self, episode, t, last):
        """Computes denominator for the formula (5.10), p. 113
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
        rho = 1.0

        # First additive
        first = 0.0
        for h in range(t + 1, last + 1):
            rho *= weight_A_S
            first += self.gamma**(h - t - 1) * rho

        rho *= weight_A_S
        # Second additive
        second = self.gamma**(last - t) * rho

        return (1 - self.gamma) * first + second

    def train_disc_aware(self, p, b, num_iter, optimal_Q):
        if DEBUG:
            # Just for now
            print("[Info] setting num_iter to 1 for debugging")
            num_iter = 1

        for _ in tqdm(range(num_iter)):
            # Per-episode inits
            new_Q = {state: {a: {"numerator": 0.0, "denominator": 0.0}
                             for a in self.env.actions}
                             for state in self.env.states}
            new_counts = {s: 0 for s in self.env.states}
            episode = self.generate_episode(b)
            last = len(episode)
            for (S, A, R) in episode:
                print("{} {} + {} yields {}".format(last, S, A, R))
                if p.decide(self.Q[S]) != A:
                   continue 
                new_counts[S] += 1
                new_Q[S]["numerator"] += self.numerator(episode, t, last)
                new_Q[S]["denominator"] += self.denominator(episode, t, last)
            print(new_Q)

def plot_history(history, title=""):
    plt.plot(history)
    plt.title(title, fontsize=20)
    plt.xlabel("Episode number", fontsize=15)
    plt.ylabel("Mean squared error, Q <-> Q*", fontsize=15)
    plt.show()

def mean_sqerror(q1, q2):
    ans = 0.0
    for state in q1:
        for action in q1[state]:
            ans += (q1[state][action] - q2[state][action])**2
    n = len(q1) * len(q1[list(q1.keys())[0]])
    return ans / n

def print_Q(Q, shape):
    for row in range(shape[0]):
        for col in range(shape[1]):
            best_action = max(Q[row * shape[1] + col].items(), key=lambda kv: kv[1])
            print("\t{}|{:.1f}".format(best_action[0][0], best_action[1]), end="")
        print("\n\n")

def eval(env, num_iter, gamma, disc_aware=False):
    global OPTIMAL_Q

    p = EpsilonGreedy(0.0)
    b = Random()
    algorithm = ROMC(env, gamma=gamma)

    # First train
    print("Evaluating optimal value function.")
    algorithm.train(p, b, num_iter, optimal_Q=None, disc_aware=disc_aware)
    optimal_Q = algorithm.Q
    if not OPTIMAL_Q:
        OPTIMAL_Q = optimal_Q

    # Second train
    print("Tracking performance.")
    history = algorithm.train(p, b, num_iter, OPTIMAL_Q, disc_aware=disc_aware)

    print("Best actions of Q:")
    print_Q(algorithm.Q, env.shape)

    #print("Showing training history.")
    #plot_history(history,
    #    "Monte Carlo (gamma={}) with\ndiscounting-{}aware importance sampling".format(
    #    algorithm.gamma, "" if disc_aware else "un"))
    return history

if __name__ == "__main__":
    params = STANDARD_RANDOM_WALK
    params["pits"] = [1, 24, 26, 67, 73]
    env = RandomWalk(**params)
    print(env)

    for gamma, marker in zip([0.9, 0.7, 0.4, 0.1], ["x", "o", ".", "v"]):
        plt.plot(eval(env, num_iter=500, gamma=gamma, disc_aware=False),
                 label="gamma = {}".format(gamma), marker=marker)
    plt.title("Monte Carlo off-policy control\nwith discounting unaware importance sampling",
              fontsize=23)
    plt.xlabel("Episode number", fontsize=15)
    plt.ylabel("Mean squared error, Q <-> Q*", fontsize=15)
    plt.legend(fontsize=20, loc="upper right")
    plt.show()
