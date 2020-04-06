from copy import deepcopy
from tqdm import tqdm

class ROMC(object):
    """Random behavior policy Off-policy Monte Carlo conrol"""
    def __init__(self, env, gamma=0.9):
        self.env = env
        self.gamma = gamma

        # Training variables
        self._init_train()

    def __repr__(self, flag=True):
        """Prints the map of best actions"""
        arrows = {"left": "<", "up": "^", "right": ">", "down": "v"}
        x, y = self.env.shape
        ans = []
        for i in range(x * y):
            ans.append("  " + arrows[max(self.Q[i], key=self.Q[i].get)])
            if (i + 1) % y == 0:
                ans.append("\n\n")
        return "".join(ans)

    def _init_train(self):
        init_prob = 1.0 / len(self.env.actions)
        init_prob = 0.0
        self.Q = {s: {a: init_prob for a in self.env.actions}
                                   for s in self.env.states}
        self.C = {s: {a: 0.0 for a in self.env.actions}
                             for s in self.env.states}

    def mean_sqerror(self, q):
        ans = 0.0
        for state in q:
            for action in q[state]:
                ans += (q[state][action] - self.Q[state][action])**2
        n = len(q) * len(q[list(q.keys())[0]])
        return ans / n

    def generate_episode(self, policy):
        """Generate episode from the given origin following the policy
        
        page 109 of the book: "all starting in the same state"
        S, A = choice(self.env.states), "empty_acton"
        """
        S = 0
        self.env.move_agent(S)

        episode = []
        while not self.env.done():
            A, prob = policy.decide(self.Q[S])
            S_prev = S
            R, S = self.env.move(A, move_agent=True)
            episode.append((S_prev, A, R, prob))
        return episode

    def importance_sampling(self, p, b, num_iter, optimal_Q):
        history = []
        for _ in tqdm(range(num_iter), desc="Episode"):
            episode = self.generate_episode(b)

            # Training inits
            G, W = 0.0, 1.0
            for (S, A, R, prob) in episode[::-1]:
                # G
                G = self.gamma * G + R
                # C
                self.C[S][A] += W
                # Q
                self.Q[S][A] += W / self.C[S][A] * (G - self.Q[S][A])

                # Break if the coverage condition is not satisfied (p. 103)
                if p.decide(self.Q[S])[0] != A:
                    break
                W *= prob

            if optimal_Q:
                history.append(self.mean_sqerror(optimal_Q))

        return history

    def importance_sampling_analytic(self, p, b, num_iter, optimal_Q):
        history = []
        for _ in tqdm(range(num_iter), desc="Episode"):
            episode = self.generate_episode(b)

            # Training inits
            G, rho = 0.0, 1.0
            numerator = [[0.0 for a in self.env.actions]
                              for s in self.env.states]
            denominator = [[0.0 for a in self.env.actions]
                              for s in self.env.states]
            for (S, A, R, prob) in episode[::-1]:
                # G
                G = self.gamma * G + R
                # Q
                numerator[S][A] += rho * G
                denominator[S][A] += rho
                self.Q[S][A] = numerator[S][A] / denominator[S][A]

                # rho
                if p.decide(self.Q[S])[0] != A:
                    break
                rho *= prob

            if optimal_Q:
                history.append(self.mean_sqerror(optimal_Q))

        return history

    def train(self, p, b, num_iter=100000, optimal_Q=None, algorithm=None):
        self._init_train()

        if algorithm == "Weighted importance sampling":
            return self.importance_sampling(p, b, num_iter, optimal_Q)
        elif algorithm == "Weighted importance sampling (analytic)":
            return self.importance_sampling_analytic(p, b, num_iter, optimal_Q)
        # elif algorithm == "Discounting-aware importance sampling":
        else:
            raise SystemExit(AttributeError("Algorithm {} is not implemented".format(algorithm)))

    def eval(self, p, b, num_avg, num_episodes, algorithm=None):
        def average_Qs(Qs):
            ans = {s: {a: 0.0 for a in self.env.actions}
                              for s in self.env.states}
            for Q in Qs:
                for s in Q:
                    for a in Q[s]:
                        ans[s][a] += Q[s][a]
            for s in ans:
                for a in ans[s]:
                    ans[s][a] /= len(Qs)
            return ans

        # Optimal
        print("Evaluating optimal value function.")
        Qs = []
        for _ in tqdm(range(num_avg), desc="Averaging run #"):
            self.train(p, b, num_episodes, optimal_Q=None, algorithm=algorithm)
            Qs.append(deepcopy(self.Q))
        optimal_Q = average_Qs(Qs)

        # Result
        print("\nTracking performance.")
        history = []
        Qs = []
        for _ in tqdm(range(num_avg), desc="Averaging"):
            history.append(self.train(p, b, num_episodes, optimal_Q, algorithm=algorithm))
            Qs.append(deepcopy(self.Q))
        self.Q = average_Qs(Qs)

        # Output
        print("Best actions of Q:")
        print(self)
        for key, values in self.Q.items():
            print("{}: {}".format(key, values))

        return [[el[i] for el in history] for i in range(len(history[0]))]
