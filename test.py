import pickle
from control import MC
from env import RandomWalk, STANDARD_RANDOM_WALK
from policy import Greedy, Soft, Random

if __name__ == "__main__":
    # Init env
    params = STANDARD_RANDOM_WALK
    params["pits"] = [5, 6, 11, 16, 22, 23]
    env = RandomWalk(**params)
    print(env)

    # Init policies for control
    p, b_soft, b_random = Greedy(), Soft(0.2), Random()
    # Init control
    control = MC(env, 0.9)

    # Train and plot
    #algorithm = "Weighted importance sampling (analytic)"
    algorithm = "Weighted importance sampling"
    algorithm = "Discounting-aware importance sampling"
    with open("log.pickle", "wb") as f:
        res = control.eval(p, b_soft, num_avg=1,
                           episodes=500000, algorithm=algorithm)
        pickle.dump(res, f)
