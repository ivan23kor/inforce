import algorithm as a
import env as e
import policy as p
env = e.RandomWalk(**e.STANDARD_RANDOM_WALK)
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
b = p.Random()          
alg = a.ROMC(env, 0.9)
l, N = 0.0, 1000
for _ in range(N):
    l += len(alg.generate_episode(b))
print(l / N)
