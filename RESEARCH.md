# Group maintenance in open space — and what it costs a UAV swarm

**Status:** all numbers below are measured output of code in this repo, not quoted from
the literature. Reproduce with the commands in [How to reproduce](#how-to-reproduce).
Where a prediction failed, the failure is reported and kept.

---

## 1. The one-paragraph version

This repo already showed that most flocking models are propped up by the periodic box
they are simulated in ([README](README.md), Experiment 1). This document extends that
argument in two directions.

**First**, cohesion is not the whole question. A flock of *several groups* can fail in a
second, independent way — the groups **fuse** into one blob — and every cohesion metric in
the literature reports a **perfect score** while it happens. Measured: segregation index
falls 1.00 → 0.50 while `largest_cluster_frac` = 1.00, `n_fragments` = 1, and group
integrity = 1.00. The groups lost no members. They lost only the thing that made them
groups.

**Second**, this is not academic. The UAV swarm controller in [`Uav/swarm_sim.py`](Uav/swarm_sim.py)
contains the line `cs_weight = 1.0/(1.0 + d**2)`, commented "Cucker-Smale-style". That is
Cucker–Smale's ψ at **β = 1**, outside the β ≤ 1/2 regime the theorem needs — and it is
then *normalised*, which makes it a DeGroot average the theorem does not describe at any β.
The controller is nonetheless operationally sound, **for a reason its own comment does not
name**. The danger is the obvious fix: making it a "proper" Cucker–Smale sum while leaving
β = 1 **loses drones** — 31× team expansion, and a displaced drone departing to 1.4 km and
still going. Half-adopting the theory is far worse than ignoring it.

---

## 2. Two ways a group dies

Every model in `models/alignment.py` and `models/cohesive.py` asks a one-group question:
did the flock hold together? Real collectives — birds, and drone teams — must also stay
**distinguishable** from each other. That is strictly stronger, and it fails in two
directions:

| failure | what happens | caught by |
|---|---|---|
| **fission** | a group loses its own members | `group_integrity`, `group_expansion` |
| **fusion** | groups merge into one blob; the labels die | `segregation_index` |

**Fusion is invisible to every cohesion metric in Experiment 1.** A fused ensemble is
maximally cohesive: one component, largest fraction 1.00. This is the same failure as
"M is a liar" from the main README, one level up — and it is why `core/metrics.py` grew a
group section rather than reusing the cohesion one.

### The model: `models/grouping.py::MultiGroupFlock`

One knob per failure mode, so each can be disabled alone:

```
dv_i/dt =  (lam_in/n_g) sum_{j in same group} psi(r_ij)(v_j - v_i)   [1] anti-fission
         -  w_sep  sum_{r_ij < r_sep}          d_ij/r_ij^2           [2] separation
         -  w_seg  sum_{j not in group, r<r_seg} d_ij/r_ij^2         [3] anti-fusion
         +  w_global (x_com - x_i)                                   [4] shared arena
         +  k_speed (v0 - |v_i|) v_hat_i                             [5] cruise
    psi(r) = K/(sigma^2 + r^2)^beta
```

Restricted to one group, **[1] *is* the Cucker–Smale system** with N → n_g, so the theorem
applies verbatim per group: at β ≤ 1/2, velocity dispersion → 0 and relative positions stay
bounded, in unbounded space, **from any initial condition**. The theorem is imported, not
re-proved. [3] is the anti-fusion term and has no theorem — whether groups survive *each
other* is measured, not asserted.

---

## 3. Experiment 2 — group maintenance without walls

`python experiments/exp2_grouping.py` — N=150, K=3 groups, **3D**, t=150, 2 seeds.

### Part A — maintenance in a shared arena

| arm | BC | S | I | M_g | defect | largest | frag | verdict |
|---|---|---|---|---|---|---|---|---|
| full (β=0.4, seg on) | periodic | **1.00** | 1.00 | 0.99 | 0 | 0.33 | 3.0 | MAINTAINED |
| full (β=0.4, seg on) | **open** | **1.00** | 1.00 | 0.99 | 0 | 0.33 | 3.0 | **MAINTAINED** |
| no segregation | periodic | **0.50** | 1.00 | 0.99 | 0 | **1.00** | **1.0** | PARTIAL |
| no segregation | **open** | **0.49** | 1.00 | 0.99 | 0 | **1.00** | **1.0** | PARTIAL |
| full + switching | periodic | **1.00** | 1.00 | 0.99 | **1** | 0.34 | 3.0 | MAINTAINED |
| full + switching | open | **1.00** | 1.00 | 0.99 | **0** | 0.33 | 3.0 | MAINTAINED |
| no seg + switching | periodic | 0.72 | 1.00 | 1.00 | **118** | 1.00 | 1.0 | PARTIAL |
| no seg + switching | open | 0.85 | 1.00 | 1.00 | **78** | 1.00 | 1.0 | PARTIAL |

`S` = segregation index (1 = distinct, 0 = fused). `I` = group integrity. `largest`/`frag`
= Experiment 1's cohesion metrics, shown for contrast.

**Read three ways:**

1. **The headline.** The `no segregation` row scores `largest` = 1.00 and `frag` = 1 — a
   flawless result by Experiment 1's standards — for a run in which the groups **merged**.
   `I` = 1.00 too: not one member lost. Cohesion metrics cannot see fusion, at all.
2. **Group maintenance survives open boundaries**, which is the positive result: S = 1.00,
   I = 1.00 with no walls, because the in-group mechanism is Cucker–Smale rather than a
   finite-range attraction that empties out as the group spreads. Note the periodic and
   open rows are *identical* — for this model the walls are doing nothing, which is exactly
   the property Experiment 1 found almost nothing else has.
3. **Segregation is what makes allegiance stable.** With switching on: **0–1 defections of
   150** with segregation, **78–118 without** — two orders of magnitude. Segregation keeps
   each agent's local majority in-group, which removes the pressure to defect, which
   preserves the groups doing the segregating.

### Two predictions that failed, and are more interesting than the ones that held

- **Fusion does not go to S = 0. It plateaus at ≈0.50** (stable from t≈40 to t≈240, and the
  same with walls as without). In-group velocity consensus leaves group-mates co-moving and
  therefore weakly co-located even inside a merged blob. Alignment buys *some* sorting for
  free — never enough to call them groups.
- **`no seg + switching` scores HIGHER purity (0.72–0.85) than `no seg` frozen (≈0.50)** —
  while **most of the swarm changes allegiance** (78–118 of 150). Defection re-sorts the
  labels onto whatever clumps exist, so purity rises *because* the labels have given up
  organising the geometry and started tracking it. **S alone calls that run better.** S plus
  the defection count calls it what it is. It is the "M is a liar" trap again, and it caught
  me.

### Part B — the β threshold, per group

Arena term **off**, so nothing but the in-group weight holds a group together. Metric:
per-group R_g growth — because that is what the theorem actually promises.

| β | start | R_g(0) | R_g(T) | **growth** | regime |
|---|---|---|---|---|---|
| **0.4** | tight | 1.48 | 1.62 | **1.09×** | unconditional (proved) |
| 0.9 | tight | 1.48 | 2.05 | **1.38×** | conditional only |
| **0.4** | spread | 5.77 | 5.77 | **1.00×** | unconditional (proved) |
| 0.9 | spread | 5.77 | 8.69 | **1.50×** | conditional only |

Note the **shape**, not just the ordering: β=0.4 does not expand from *either* start, while
β=0.9 does *worse the worse its start is*. That asymmetry is exactly what "unconditional"
versus "conditional" means, and it is why β is not a tuning knob.

> **Why not score this with `group_integrity`?** Because CS does not promise it. The theorem
> gives *bounded* relative positions — bounded, not small, and not "within r_link". A group
> that starts spread over 5 units stays spread over 5 units: growth 1.00, a flawless result,
> while integrity at r_link=1.5 reads 0.07 because it was never connected at that range to
> begin with. Integrity would be scoring the initial condition and calling it a model
> failure. **Measure what the theorem promised.**

### A bug worth reporting, because it is the repo's own thesis biting back

An earlier version of this table reported the periodic `no segregation` arm at **S = 0.26**
against 0.49 open, and I nearly wrote a paragraph explaining why walls make fusion worse.
They do not. The arena term [4] was attracting agents to `positions.mean(axis=0)` — and on
a torus the naive mean of coordinates **is not a point on the flock**: agents at x=0.1 and
x=L−0.1 are neighbours and average to the far side of the box. The "force" was an artifact
of the wrapping. Fixed to use the circular-mean `centroid()` (correct under every boundary),
and the periodic and open arms now agree at ≈0.50, which is the sensible answer: **fusion
does not care about the walls.**

The irony is the point. This repo exists to argue that periodic boundaries quietly
manufacture results, and a periodic boundary quietly manufactured one *here*, in the model
written to make that argument. Every model in this repo obtains separations through
`boundary.displacement()` — that discipline is what makes the BC swap valid — and this term
reached around it for a centroid. **The discipline only protects the code that follows it.**

### A method note worth more than the numbers

β > 1/2 does **not** cause fission when the arena term is on, or from a tight start. Both
arms score MAINTAINED. The theorem is about what is *guaranteed*; a favourable initial
condition satisfies conditional flocking's condition, and any cohesive force that isn't
Cucker–Smale will mask the threshold entirely. Part B exists to strip that help away. **An
experiment that cannot fail cannot discriminate**, and the first version of Part B could not.

---

## 4. 3D visualization

`core/viz3d.py` renders any model in the repo. The point is not decoration: the entire
argument is that a scalar can look healthy while the flock it describes has evaporated.
`M = 0.556` and "it shattered into 5 pieces" are **the same frame** — every movie burns the
live metrics into the HUD so the picture and the numbers cannot drift apart.

```bash
python experiments/viz3d_demo.py --list
python experiments/viz3d_demo.py --scene fusion --time 200
```

| scene | what you watch |
|---|---|
| `collapse` | Vicsek on a torus, then the *same model* in open space. Measured in-frame: M=0.983, frag=1 on the torus; M=0.556, frag=5, largest=0.44 without walls. |
| `survives` | Cucker–Smale β=0.4, open. The control that refuses to spread. |
| `groups` | `MultiGroupFlock`, 3 groups, open. Maintenance. |
| `fusion` | **the one to watch.** Segregation off: the groups merge while the HUD's cohesion metrics read a perfect 1.00 throughout. |

Two visualization choices are load-bearing and both are honesty constraints, not aesthetics:

- **Fixed zoom, moving camera** (`follow=True`). Autoscaling per frame would renormalise
  away the group's expansion — an evaporating flock would look like a stable one that
  merely drifts. Fixed zoom means dispersal looks like dispersal.
- **Trails are cut at periodic wraps.** A bird leaving at x=L and re-entering at x=0 moved
  ~0, but its coordinates jumped by L; drawing that segment paints a line it never flew.

**A density trap worth flagging.** The 3D scenes use L=6, not Experiment 1's L=10. Re-using
L=10 in 3D gives ρ=0.12 and ~1.7 neighbours inside r_link: the flock starts as **43
disconnected fragments before a single step**. That scene looks like a dramatic collapse
and is measuring *dilution*. L=6 restores ~8.0 neighbours, matched to the 2D experiment on
the quantity the interaction actually depends on, so topology is the only thing left varying.

---

## 5. The UAV scenario — where this stops being about birds

[`Uav/swarm_sim.py`](Uav/swarm_sim.py) flies 12 drones in a 2D V-formation behind one
leader, and reports order 0.95–1.0 across all four of its scenarios. Those numbers are
real. They are also produced by a stack in which the **formation term** — every follower
pulled toward `leader_pos + offset` — is doing the cohering. That is **one drone**, and
drones fail.

[`Uav/multi_team_3d.py`](Uav/multi_team_3d.py) rebuilds it as **K teams in 3D open
airspace** with crossing waypoints (deconfliction), per-drone uncorrelated turbulence, a
hard safety floor, and a `leader_loss_at` event. Kill the leader and the alignment layer is
all that remains — which is when the exponent in `1.0/(1.0 + d**2)` stops being a detail.

> **The airspace is open. That is not a modelling choice — it is the truth.** Every result
> in Experiment 1 about models that only work on a torus applies here with no translation.

### Test 1 — leader loss under turbulence (t = 400 s, ~6.5 min sortie)

| alignment rule | theorem? | R_g growth | integrity | M_team | min_sep | verdict |
|---|---|---|---|---|---|---|
| `swarm_sim.py` rule (β=1, **normalised**) | no | **1.01×** | 0.94 | 1.00 | 2.61 | **HOLDS** |
| β=1.0, CS sum (unnormalised) | no | **30.83×** | **0.17** | 0.45 | 2.52 | **SCATTERED** |
| β=0.5, CS sum | **yes** | **1.02×** | 1.00 | 1.00 | 2.53 | **HOLDS** |
| β=0.4, CS sum | **yes** | **1.01×** | 1.00 | 1.00 | 2.53 | **HOLDS** |

Both survivors hold — **for different reasons, with different guarantees**. β≤0.5 holds *by
theorem*. The current rule holds because normalising keeps the coupling strong at any range.

> **Run length is load-bearing.** At t=120 s every arm scored 1.01× and this table read
> "all safe". Dispersal is a **rate**: the β=1 leak needed ~400 s of flight to show itself.
> A short sortie in sim is not evidence of a safe one.

### Test 2 — the gust / straggler event

One drone blown D metres out of formation with a velocity error, after leader loss. Does its
velocity re-converge (**gap bounded** → trackable) or not (**gap grows** → lost airframe)?

| alignment rule | D | gap @100 s | gap @600 s | outcome |
|---|---|---|---|---|
| `swarm_sim.py` rule (β=1, normalised) | 20 / 60 / 120 | 22 / 63 / 123 | **22 / 63 / 123** | bounded |
| **β=1.0, CS sum (unnormalised)** | 20 / 60 / 120 | 127 / 252 / 319 | **548 / 1202 / 1408** | **LOST, still growing** |
| β=0.5, CS sum | 20 / 60 / 120 | 23 / 65 / 129 | **22 / 64 / 127** | bounded |
| β=0.4, CS sum | 20 / 60 / 120 | 22 / 63 / 124 | **22 / 63 / 123** | bounded |

**This overturned the hypothesis the experiment was built to confirm.** The prediction was
that β=1 would lose the drone and the theorem would save it. Instead:

1. **The normalisation is what saves the current code.** A lone straggler's only neighbours
   are its own team, and a weighted average gives them full weight *however far away they
   are*, so its velocity re-converges. `swarm_sim.py` is operationally sound — by luck, not
   by theorem, and not for the reason its comment claims.
2. **The trap is the obvious fix.** Unnormalise to get a "real Cucker–Smale sum" while
   leaving β=1 and the drone is gone: 1.4 km and still departing. **Half-adopting the theory
   is much worse than ignoring it.** Both changes must land together.
3. **β ≤ 1/2 + unnormalised** gives bounded gaps *and* a guarantee.

### The caveat that matters most operationally

**No alignment rule recovers the straggler.** 22 m at t=100 s is 22 m at t=600 s. Cucker–Smale
gives velocity consensus and a bounded gap; **it never promised to close one**, and it does
not. The drone cruises alongside forever at a fixed offset. Rejoining requires the formation
layer (a live leader) or a genuine attraction term. **If your CONOPS assumes a gust-displaced
drone slides back into its slot unaided, no choice of β delivers that.**

### Recommendations for `swarm_sim.py`

1. **Do not unnormalise without also setting β ≤ 1/2.** This is the only change here that
   can make things catastrophically worse. Currently safe by accident; the "cleanup" is what
   bites.
2. **To get the guarantee**, use the unnormalised sum with β = 0.4–0.5 (`psi = K/(sigma**2 +
   d**2)**0.4`, summed, normalised by team size — not by the weight sum). Cost: one
   exponent. Benefit: unconditional flocking from any initial condition, no leader required.
3. **Fix the comment either way.** `1/(1+d^2)` is not "Cucker–Smale-style" in any regime
   once it is normalised. Someone will eventually trust that comment.
4. **The leader is a single point of failure for cohesion**, and the order parameter cannot
   warn you: while the leader lives, the swarm is cohesive for reasons that have nothing to
   do with the flocking layer. Test leader loss explicitly.
5. **Measure per team, never per ensemble.** Three deconflicted teams score
   `largest_cluster_frac` = 0.33 and global M ≈ 0 *by construction*. Both numbers are
   correct and meaningless. The first version of `run_multi_team.py` made exactly this
   mistake — the trap this repo documents is easy to fall into even while documenting it.
6. ~~`run_experiments.py` writes to a hard-coded `/home/claude/swarm_sim/`~~ — **fixed**;
   it now writes beside itself and runs anywhere. Its four scenarios still reproduce
   (order 0.948, min-sep 2.19 m, formation error 1.71 m, localization 0.148 m for D).

---

## 6. What is proved, what is measured, what is neither

Being explicit, because the point of the repo is that these get conflated:

| claim | status |
|---|---|
| Per group, β ≤ 1/2 ⇒ unconditional flocking | **Proved** (Cucker & Smale 2007), imported per group, and *reproduced* here (growth 1.00–1.09× from any start) |
| Groups + segregation survive open boundaries | **Measured** (S=1.00, I=1.00). No proof. The composite system with terms [2]–[4] has none. |
| Segregation stabilises allegiance | **Measured** (0–2 defections vs 78–92). Mechanism is plausible and stated; not proved. |
| Fusion is invisible to cohesion metrics | **Demonstrated**, and it is a fact about the metrics, not a model result |
| The normalisation saves `swarm_sim.py` | **Measured**, in this scenario, at these parameters. Not a theorem, and not a guarantee under conditions not tested. |
| A straggler is never recovered by alignment | **Measured**, and consistent with what CS proves (velocity consensus ≠ gap closure) |

**Limits.** 2 seeds — enough to establish ordering, not for finite-size scaling. The UAV
model is a *behavioural-layer* prototype with saturated velocity setpoints, not quadrotor
flight dynamics; the localization/GPS-denied layer of `swarm_sim.py` is **not** carried over
here, so these numbers assume perfect state knowledge. Turbulence is per-drone i.i.d. — real
gusts are spatially correlated, and a correlated field is *gentler* on cohesion (it displaces
a team uniformly), so the dispersal figures here are conservative in that respect and
optimistic in others. None of this has touched hardware.

---

## How to reproduce

```bash
pip install numpy scipy matplotlib pillow

python tests/test_theorems.py               # 13/13, incl. the grouping claims above
python experiments/exp2_grouping.py         # section 3 tables
python Uav/run_multi_team.py                # section 5 tables
python Uav/run_multi_team.py --viz          # + 3D movies of leader loss

python experiments/viz3d_demo.py --scene fusion     # section 4
python experiments/viz3d_demo.py --time 400         # longer runs
```

Every claim in this document is one command away from being falsified. That is the intent.
