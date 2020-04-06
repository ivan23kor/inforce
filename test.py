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

    # Train and plot
    algorithm = "Weighted importance sampling"
    with open("log", "w") as f:
        res = control.eval(p, b_epsilon_greedy, num_avg=2,
                           num_episodes=1000, algorithm=algorithm)
        f.write(str(res))
