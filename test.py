import matplotlib.pyplot as plt

from control import ROMC
from env import RandomWalk, STANDARD_RANDOM_WALK
from policy import Greedy, EpsilonGreedy, Random

if __name__ == "__main__":
    # Init env
    params = STANDARD_RANDOM_WALK
    params["pits"] = [5, 6, 11, 16, 22, 23]
    env = RandomWalk(**params)
    print(env)

    # Init policies for control
    p, b_epsilon_greedy, b_random = EpsilonGreedy(0.2), EpsilonGreedy(0.5), Random()
    # Init control
    control = ROMC(env)
    # control.generate_episode(b_greedy)
    # exit(0)

    # Train and plot
    # for gamma, marker in zip([0.9, 0.7, 0.4, 0.1], ["x", "o", ".", "v"]):
    algorithm = "Weighted importance sampling (analytic)"
    algorithm = "Weighted importance sampling"
    for gamma, marker in zip([0.1], ["x"]):
        control.gamma = gamma
        plt.plot(control.eval(p, b_epsilon_greedy, num_avg=2, num_episodes=50000,
                              algorithm=algorithm),
                 label="gamma = {}".format(gamma), marker=marker, markersize=4)
    plt.title("Monte Carlo off-policy control\n {}".format(algorithm),
              fontsize=23)
    plt.xlabel("Episode number", fontsize=15)
    plt.ylabel("Mean squared error, Q vs Q*", fontsize=15)
    plt.legend(fontsize=20, loc="upper right")
    plt.show()
