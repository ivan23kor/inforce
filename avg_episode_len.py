import env as e
from control import ROMC
from policy import Random
env = e.RandomWalk(**e.STANDARD_RANDOM_WALK)
print("Testing on the board:")
print(env)
"""
>>> env
^  .  .  .  .  .  .  .  .  .
.  .  .  .  .  .  .  .  .  .
.  .  .  .  .  .  .  .  x  .
.  .  x  x  .  .  .  .  .  .
.  .  .  .  .  .  .  .  .  .
.  .  .  .  .  .  x  .  .  .
.  .  x  .  .  .  .  .  .  .
.  .  .  .  .  .  .  .  .  .
.  .  .  .  .  .  .  .  .  .
.  .  .  .  .  .  .  .  .  o
"""
b = Random()          
control = ROMC(env, 0.9)
l, N = 0.0, 1000
for _ in range(N):
    l += len(control.generate_episode(b))
print("Average episode length is {}".format(l // N))
