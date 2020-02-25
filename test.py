from algorithm import OffPolicyMC
from env import Env, ACTIONS
from policy import Greedy, Random

def func_info(func, *args):
    def new_func(*args):
        print("{} |\n{}".format(func.__name__, "-" * (len(func.__name__) + 2)))
        func(*args)
        print("-" * 80, end="\n\n")
    return new_func

@func_info
def test_env():
    env = Env((10, 10), [1, 0], [[0, 1], [0, 3], [8, 9]])
    def print_act(R, S):
        print("R: {}, S': {}".format(R, S))
    print_act(*env.act([0, 0], "left"))
    print_act(*env.act([0, 0], "right"))
    print_act(*env.act([0, 0], "down"))

@func_info
def test_algorithm():
    env = Env((3, 3), [2, 2], [[0, 1]])
    actions = env.actions
    states = env.states
    algorithm = OffPolicyMC(actions, states, env)
    algorithm.train()

@func_info
def test_random(n=10):
    env = Env((10, 10), [1, 0], [[0, 1], [0, 3], [8, 9]])
    actions = env.actions
    states = env.states
    Q = [[0, 0, 1, 0] for _ in states]
    random_p = Random(actions)
    for _ in range(n):
        print(random_p.decide(Q, [0, 0]))

if __name__ == "__main__":
    # test_env()
    test_algorithm()
    # test_random(7)
