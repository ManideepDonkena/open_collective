"""
example.py  --  the simplest possible run. It makes a picture of a flock.

WHAT TO DO
  1. First run "start.py" once (that installs everything).
  2. Then run this file. It saves a picture called "flock.png" in this folder.
     Open flock.png to see the birds and which way they are pointing.

You can change the three numbers marked below and run it again to see what
happens. That's it -- no other knowledge needed.
"""

import sys
from pathlib import Path

# make sure Python can find the project's code, however you run this file
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")           # just save a file; no pop-up window needed
import matplotlib.pyplot as plt

from core import make_boundary, run
import core.init as cinit
from models import VicsekModel

# ---- the three numbers you can play with ----
NUMBER_OF_BIRDS = 200           # try 50 or 500
NOISE           = 0.2           # 0.0 = tidy flock, 1.0 = messy/random
STEPS           = 300           # how long to let it fly
# ---------------------------------------------

world = make_boundary("open", dim=2)                       # an open space
birds = cinit.random_init(NUMBER_OF_BIRDS, world, speed=0.5)  # scatter the birds
rule  = VicsekModel(world, r_max=1.0, eta=NOISE)           # the flocking rule
final, history = run(rule, birds, steps=STEPS, dt=0.05, r_link=1.0)

# draw the result: a dot + a small arrow for each bird
pos, direction = final.positions, final.headings
plt.figure(figsize=(6, 6))
plt.quiver(pos[:, 0], pos[:, 1], direction[:, 0], direction[:, 1],
           color="steelblue", pivot="mid")
plt.title(f"A flock of {NUMBER_OF_BIRDS} birds (noise = {NOISE})")
plt.axis("equal")
plt.axis("off")
plt.savefig("flock.png", dpi=120, bbox_inches="tight")

# how "together" is the flock? 1.0 = all pointing the same way, 0 = all mixed up
togetherness = history["polar_order"][-1]
print("Saved flock.png -- open it to see your flock!")
print(f"Togetherness score: {togetherness:.2f}  (1.0 = perfectly aligned)")
