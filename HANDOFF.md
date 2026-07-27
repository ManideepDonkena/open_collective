# HANDOFF — Flocking Models vs. Biological Reality

**Purpose:** self-contained context to resume this work in a fresh chat.
**Status:** codebase built and validated; core criticism established; **direction just
changed** — see §5. Read §0 and §5 first.

---

## §0. TL;DR for the next session

We started by asking whether two papers (in project files: `PAPER_1.pdf`, `PAPER_2.pdf`)
survive outside a periodic box. **They don't.** We built a codebase proving it.

Then the user asked the question that broke the frame:

> *"How can we say Cucker–Smale works? A bird doesn't have information of all the birds.
> And Paper 2 is doing the same thing."*

**This was correct and it invalidated my proposed fix.** Cucker–Smale is all-to-all —
biologically absurd. I had been praising it for surviving open space while attacking
Paper 2 for unphysical coupling, without noticing both use unbounded-range current-state
information.

**New goal (user's words):** *not only consensus — check whether it's feasible in real
life.* Model what a bird can **actually sense**. Ground in starling data.

**The literature already answers this**, and neither paper cites it: **Pearce, Miller,
Rowlands & Turner, PNAS 111, 10422 (2014)**. Birds get long-range information **without
knowing any far bird's position or velocity** — from the **projected silhouette** of the
flock against the sky. That is the resolution. See §3.

---

## §1. The two papers

Both by **Jyotiranjan Beuria** (IIT Mandi / IKS Research Centre, ISS Delhi).

**PAPER_1** — *Collective motion from quantum-inspired dynamics in visual perception*
(Beuria, Chaurasiya & **Behera**; arXiv 2409.18985v4, Jun 2025)
- Bird's "follow / don't follow" decision modelled as a quantum superposed/entangled
  state over neighbours in a **vision cone** (angular width α = π/2, radii [r_min, r_max]).
- A Hermitian **perception operator**; force = its quantum expectation value.
- Reduces to Vicsek for positive-coefficient states (Φ+ → ½, GHZ3 → ⅓, W3 → 9/4, uniform → 3/2).
- **Φ− (negative coefficient) → destructive interference → no order** (their Eq. 32).
- Sim: N=200, 2D **periodic** box L=10, η=0.2, α=π/2, r∈[0.1, 5].

**PAPER_2** — *Non-Markovian Collective Motion from Self-Regulated Perceptual Dynamics*
(Beuria alone; dated May 11, 2026)
- Each agent: **fast perceptual register** (Bloch vector per channel, GKSL-derived) +
  **slow regulatory scalar `s_i`** integrating alignment history and feeding back.
- Explicitly **not** a quantum claim — GKSL used only as a positivity-preserving
  description of bounded two-state variables.
- Memory emerges from an **enlarged Markovian state space**: full state is Markovian;
  hide `s_i` and the reduced dynamics become non-Markovian. No imposed memory kernel.
- Results: slow–fast relaxation, hysteresis under cyclic λ_fb, non-monotonic M–S coupling.
- Sim: N=400, 2D **periodic** domain, **fixed topological neighbour graph**, w_ij = 1/n.

**Assessment:** competent mid-tier work. PAPER_2 is notably more mature than PAPER_1
(drops the speculative quantum framing, careful about claims). Neither is field-defining.
Benchmarks they sit against: Vicsek 1995; Toner–Tu; Grégoire–Chaté; Ballerini/Cavagna
empirically; **inertial spin model** (Attanasi et al.) as the direct competitor to PAPER_2.

---

## §2. What we built and what it showed

**Codebase:** `open_collective/` (also zipped). 16 models. Boundary condition is a
**swappable first-class object** (`PeriodicBoundary` / `OpenBoundary` / `ReflectingBoundary`).
Every model gets separations only via `boundary.displacement()` — that discipline is what
makes the periodic↔open swap valid.

```
core/      boundary.py  neighbors.py  metrics.py  base.py
models/    alignment.py (Vicsek, PerceptionQuantum=PAPER_1, SlowFastPerception=PAPER_2)
           cohesive.py  (Boids, Couzin, D'Orsogna, Cucker-Smale, Olfati-Saber)
           formation.py (Displacement, Distance/rigid, Leader-follower, Cyclic pursuit)
           consensus.py (DeGroot, Friedkin-Johnsen, SignedFJ, Altafini, GroupConsensus)
tests/     test_theorems.py   -> 10/10 passing
experiments/ exp1_boundary_collapse.py, exp1_results.py
```

### Headline measurement (N=120, L=10, 1500 steps, identical ICs, only topology differs)

| model | M (periodic) | M (open) | R_g growth | fragments | largest |
|---|---|---|---|---|---|
| Vicsek (metric) | 0.989 | 0.843 | 4.41× | 3 | 0.76 |
| Vicsek (topological k=7) | 0.990 | 0.768 | **3.00×** | 6 | 0.65 |
| **PAPER_1 [Φ+]** | 0.987 | 0.434 | **8.10×** | **39** | **0.17** |
| **PAPER_1 [GHZ3]** | 0.987 | 0.397 | **8.23×** | **42.5** | **0.23** |
| **PAPER_2** | 1.000 | 0.695 | 3.89× | 18.5 ⚠️ | 0.60 |
| Boids | 0.602 | 0.508 | 2.05× | 3 | 0.77 |
| Couzin | 0.723 | 0.293 | 4.23× | 5 | 0.47 |
| Olfati-Saber | 0.999 | 0.393 | 3.56× | 17 | 0.31 |
| Cucker-Smale β=0.9 | 1.000 | 0.927 | 1.88× | 40.5 | 0.38 |
| **Cucker-Smale β=0.4** | 1.000 | **1.000** | **1.03×** | 3 | **0.97** |
| **D'Orsogna (catastrophic)** | 0.534 | 0.119 | **0.31×** | **1** | **1.00** |

### Established findings (these survive the reframe)

1. **The order parameter M is not sufficient and can be actively misleading.** Vicsek
   scores 0.843 in open space while already split into 3 non-communicating flocks. M
   averages over groups that cannot see each other. **Never report M without fragmentation.**
2. **PAPER_1 is the worst performer in the study — worse than 1995 Vicsek.** Its forward
   vision cone means a bird that falls behind has the flock *behind its own cone* and can
   never recover contact. **The paper's signature feature is what destroys it.** (But see
   §3.4 — the cone may also be biologically backwards.)
3. **Neither paper has an attraction/cohesion term.** Only alignment. Periodic walls
   supply the cohesion for free.
4. **Cucker–Smale's β = 1/2 threshold is visible in simulation** (0.4 → 1.03× and intact;
   0.9 → 1.88× and 40 fragments). Theorem reproduced.
5. **Cohesion is a parameter regime, not a model property.** D'Orsogna's *H-stable* regime
   (C·l² > 1) **disperses** (6.2×, 60 fragments); its cohesive mills live in the
   *catastrophic* regime (C·l² < 1, measured 0.31× — it contracts). I asserted the
   opposite in a test; **the code corrected me.**

### ⚠️ Corrections I made — carry these forward

- **"PAPER_2 breaks into 19 fragments" was a bad claim.** It averaged 2 seeds over
  **bimodal** behaviour. Per-seed fragments: **1, 36, 5, 34**. Sometimes it survives
  completely; usually it shatters. The average describes no actual run. *I committed the
  same sin I criticised the papers for.* **All numbers in the table above need re-running
  with ≥10 seeds, reported as median + IQR.** This is the top TODO.
- **"PAPER_2 doesn't know it's broken" was unfair.** Beuria states it as **limitation #1**
  in his Discussion: the fixed topological graph "isolates the effect of the internal
  slow–fast feedback loop from changes in neighbour identity, but it does not yet capture
  all spatial effects of a fully moving active-matter system with dynamically updated
  neighbourhoods." He names it. The fair criticism is narrower: **he identifies the
  limitation but doesn't measure its cost** — and the cost is that his headline observable
  becomes meaningless in the regime he wants to claim.

### The frozen-graph diagnostic (answers a specific user question)

PAPER_2 Eq. (25): `u_i(t) = ρ p̂_i(t) + Σ_j w_ij (1+m_z^{ij}(t))/2 · p̂_j(t)`

**`p̂_j(t)` is the neighbour's CURRENT heading.** So: neighbour **identity** is frozen at
t=0; neighbour **values** are live. That is *not* a delay or memory effect — it is
**instantaneous infinite-range coupling to a fixed set of six partners.** Measured:

| | seed 1 | seed 3 |
|---|---|---|
| reported M | 0.219 | **0.992** |
| physical fragments | 27 | **33** |
| frozen-partner distance, t=0 → end | 0.96 → mean 4.02 / **max 48.8** | 0.96 → mean 2.72 / **max 30.8** |
| % of "neighbours" out of sight range (1.5) | **57.2%** | **57.5%** |

**Seed 3 is the smoking gun:** M = 0.992 with the flock in 33 pieces. The fragments stay
aligned *because they never stop talking*. The frozen graph is the direct mechanism that
makes M lie.

**Where PAPER_2's memory actually comes from:** the slow register `s_i` (own history,
horizon 1/γ_s) — **not** the graph. **The non-Markovian result survives this criticism.**
What doesn't survive is the spatial interpretation.

---

## §3. THE REFRAME — what starlings actually do

Sources: STARFLAG project / Cavagna–Giardina group (Rome), + Turner group (Warwick).

### 3.1 Ballerini et al., PNAS 105, 1232 (2008) — topological, not metric

- 3D reconstruction of flocks of a few thousand (e.g. 1,246 birds at ~70 m, ~11 m/s).
- **Each bird interacts with a fixed number of neighbours: 6–7** — *not* all birds within
  a radius. Interaction depends on **topological** distance (1st, 2nd, 3rd neighbour), not
  metric distance.
- **Their argument:** topological interaction is **indispensable for cohesion** against the
  large density changes caused by predation. Under simulated predator attack, **metric**
  models fission and shed stragglers; **topological** models almost never do.
- Also found: **anisotropy** in nearest-neighbour angular distribution.

**Cross-check against our data:** topological Vicsek gave **3.00×** vs metric **4.41×**.
Topological *is* better for cohesion — Ballerini confirmed. **But it still disperses.**

### 3.2 The dispersal is a known theorem, not our discovery

**Ginelli & Chaté:** metric-free (topological) models **support a zero-density steady
state — diffusive expansion continues indefinitely.** So "use topological neighbours"
does *not* solve open-space cohesion, and the field knows it. Our 3.00× measurement is a
rediscovery. **Cite Ginelli & Chaté, don't claim novelty here.**

### 3.3 Pearce, Miller, Rowlands & Turner, PNAS 111, 10422 (2014) — **the answer**

This paper directly addresses the user's objection and neither Beuria paper cites it.

- **Their claim:** *local interactions alone are **insufficient** to explain the
  organisation of large flocks, and the mechanism for the long-range information exchange
  needed to control density **remains unknown**.*
- **Their mechanism — projection.** Coarse-grain each bird's visual input to a pattern of
  **dark bird against light sky**. Birds fly toward the resolved vector sum of the
  **domain boundaries** of that projected view.
- **This is the resolution to the user's objection.** A bird gets **long-range
  information** while knowing **no far bird's position, velocity, or identity**. It only
  reads its own retina. Local sensing, global information. Cucker–Smale's all-to-all sum
  is not needed and not what birds do.
- **Emergent result: marginal opacity.** Flocks self-assemble to the maximum density at
  which a typical bird can still just see out in many directions.
- **Measured:** 118 uncorrelated UK flock measurements, opacity **μ = 0.30, σ² = 0.059**;
  public-domain images **μ = 0.41, σ² = 0.012**. Uniform-distribution null rejected at
  **99.99%**. Flocks are marginally opaque — they need not have been; they could have been
  diffuse or fully opaque.
- **Also gives faster information transfer** than local models, and **density control** —
  the thing metric-free models lack.
- **Follow-up (Lewis & Turner):** edge birds need an **inward** motional bias; interior
  birds need an **outward** bias. Depth-in-flock is inferable from the visual environment.
  This produces a **finite spatial extent steady state** — exactly what Ginelli–Chaté lack.

### 3.4 ⚠️ PAPER_1's vision cone may be biologically backwards — **VERIFY THIS**

PAPER_1 uses a **90° forward cone** (α = π/2). Ballerini's measured **anisotropy** shows
nearest neighbours are preferentially **lateral**, with a deficit along the direction of
motion. Starlings have lateral foveae and near-panoramic vision; the blind spot is
**behind and narrow**, not a forward 90° window.

**If so, PAPER_1's cone is close to the inverse of the biology** — which would explain why
it fragments 39 ways (see §2 finding 2). **This is a hypothesis, not yet verified.**
Next session: get the actual starling visual field (~300°?) and Ballerini's anisotropy
figure before asserting it.

### 3.5 Attanasi et al., Nature Physics 10, 691 (2014) — PAPER_2's real competitor

- Tracked up to 400 starlings through **collective turns**.
- **Information about direction changes propagates with a LINEAR dispersion law and
  NEGLIGIBLE attenuation** ("superfluid transport"). Minimises group decoherence.
- **This contrasts starkly with existing models, which predict DIFFUSIVE transport.**
- Requires **behavioural inertia** → the **inertial spin model**. Cavagna et al. (2025)
  improved it with nonlinear torques to match experiment.

**Implication for PAPER_2:** this is the *empirically calibrated* memory model for
starlings. PAPER_2 acknowledges the comparison and correctly notes its slow variable is
**not conjugate to heading** (unlike the inertial spin), acting instead as a regulatory
bias producing hysteresis. **So PAPER_2 must show its memory produces a signature the
inertial spin model does not** — otherwise it is redundant for birds. It may still be the
right model for **robot swarms** (adaptive gain) where inertial spin has no motivation.

### 3.6 Other empirical anchors

- **Cavagna et al., PNAS 107, 11865 (2010):** scale-free correlations — correlation length
  scales with flock size. No intrinsic length scale.
- **Young et al., PLoS Comput Biol (2013):** *why seven?* Under **sensing uncertainty**,
  6–7 neighbours optimises the trade-off between group cohesion and individual effort.
  This is the information-cost argument the user is reaching for.
- **Bialek et al., PNAS 109, 4786 (2012):** maximum-entropy / statistical mechanics for
  natural flocks.
- Murmurations reach **~300,000 birds**.

---

## §4. Restating the real-life scenario

**What a starling HAS:**
- ~6–7 tracked neighbours, **topological** (Ballerini 2008)
- **A coarse-grained silhouette of the whole flock** on its retina — long-range info with
  no far-bird identity (Pearce 2014)
- Near-panoramic vision, lateral foveae (⚠️ verify)
- **Behavioural inertia** — cannot turn instantly (Attanasi 2014)
- A **roost** — murmurations have a destination, i.e. a real global attractor
- Finite duration — murmurations are **transient** (~20–40 min? ⚠️ verify).
  *Open question: is the dispersal timescale even relevant on the murmuration timescale?*
- Selection pressure: **peregrine attacks** — the reason cohesion is non-negotiable

**What a starling does NOT have:**
- Position/velocity of far birds → **kills Cucker–Smale as biology**
- A frozen neighbour list → **kills PAPER_2's simulation setup**
- A periodic box → **kills both papers**
- A global reference frame, GPS, or infinite time

### Scorecard under the new criterion

| model | mechanism physically available to a bird? | holds together in open space? |
|---|---|---|
| Vicsek (metric) | ✅ yes | ❌ no |
| Vicsek (topological) | ✅ yes (Ballerini) | ❌ no (Ginelli–Chaté) |
| **PAPER_1** | ⚠️ cone may be inverted | ❌ worst in study |
| **PAPER_2** | ❌ frozen graph + infinite range | ⚠️ bimodal |
| Couzin | ✅ **most biological** | ❌ evaporated (4.23×) |
| Cucker–Smale | ❌ **all-to-all — absurd** | ✅ yes (1.03×) |
| D'Orsogna | ❌ Morse potential, not a sense | ✅ yes (0.31×) |
| **Pearce–Turner projection** | ✅ **yes — retina only** | ✅ **yes + density control** |

**Nothing in our codebase currently occupies the top-right quadrant. Pearce–Turner does.
That is the model to implement and beat.**

**The honest headline is no longer "Paper 1 fails."** It is:

> **No model we tested is both biologically feasible and cohesive in open space.**
> Cucker–Smale buys its theorem with telepathy. PAPER_2 buys stability with a frozen
> graph. Couzin is the most biological and it evaporates. The one candidate that is both
> — visual projection to marginal opacity — is absent from both Beuria papers and from
> our code.

### ⚠️ Kill the old "fix"

I previously proposed bolting Cucker–Smale's ψ(r) onto PAPER_1's perception operator to
"inherit the theorem for free." **This is dead.** It imports all-to-all coupling, which
contradicts PAPER_1's own founding premise (birds have *limited neural resources*, hence
a *narrow* vision cone). Grafting infinite-range summation onto a model built on
perceptual scarcity is incoherent. **A real fix needs finite range AND cohesion AND
dispersal robustness simultaneously — an open problem, not an edit.**

---

## §5. Next steps, in priority order

1. **Implement `ProjectionModel`** (Pearce–Turner) in `models/cohesive.py`. Coarse-grain
   each agent's view into angular bins → dark/light → steer toward the resolved vector sum
   of boundaries. Add an **opacity Θ** observable to `core/metrics.py`. **Success test:
   does it self-assemble to Θ ≈ 0.30 in open space, unforced?** That would be the first
   model in the repo in the top-right quadrant.
2. **Re-run exp1 with ≥10 seeds, report median + IQR.** The 2-seed averages are not
   defensible and PAPER_2 is bimodal. *Do this before quoting any number to anyone.*
3. **Verify §3.4** — starling visual field and Ballerini's anisotropy. If PAPER_1's
   forward cone is inverted relative to the biology, that is a clean, publishable,
   *constructive* criticism: **swap the cone to lateral/panoramic and re-measure.**
   Predicted: fragmentation drops sharply. **This is the highest-value single experiment.**
4. **Add a `RoostAttractor`** (a fixed point in space) and ask the reframed question:
   *given a roost, does open-space dispersal even matter on a 30-minute timescale?*
   This may dissolve the whole problem for birds while leaving it real for robot swarms.
5. **Implement the inertial spin model** as PAPER_2's benchmark. Test: does PAPER_2's slow
   register produce **linear undamped** information propagation (Attanasi's measurement),
   or **diffusive** (what ordinary models give)? If diffusive, PAPER_2 does not describe
   starlings — but may still describe robots.
6. **Information-cost axis (the user's actual interest).** Build a table: bits/second each
   model demands per agent. Cucker–Smale = N×(position+velocity)/step. Topological = 7×.
   Projection = one coarse retina. Cross-reference **Young et al. 2013** on why 7 is optimal
   under sensing noise. **This is the "is it feasible in real life" question made
   quantitative, and it is a paper on its own.**

---

## §6. The Tripathy / Shrinate bridge (separate thread, still live)

- **Twinkle Tripathy** (Asst. Prof., EE, IIT Kanpur) + **Aashi Shrinate** (PhD, PMRF).
  Control theory: signed Friedkin–Johnsen, multiconsensus, echo chambers, influence
  centrality, cyclic pursuit.
- **Institutional link:** Shrinate, Tripathy & **Behera** have a 2026 preprint on echo
  chambers under the FJ model. **Behera co-authors PAPER_1.** The groups are one handshake
  apart and have never cited each other.
- **Proven in code** (`tests/test_theorems.py::test_beuria_phi_minus_is_a_signed_network`):
  PAPER_1's Φ− state = **all-negative complete graph** = **structurally unbalanced** →
  **Altafini's theorem predicts collapse.** Measured |x| → 5.8e-22. His numerical
  observation now has a proof.
  - **The regime PAPER_1 never tested:** make the signed perception graph **balanced**
    (3 positive, 3 negative) → **not disorder** but **bipartite consensus**:
    `[-0.0154 ×3, +0.0154 ×3]`, camps matching the gauge partition exactly.
  - **Therefore PAPER_1's conclusion is too strong.** Correct statement: *negative
    perceptual coefficients destroy cohesion **iff** the induced signed perception graph
    is structurally unbalanced. If balanced → two counter-propagating sub-flocks.*
  - **This is a new, falsifiable prediction the paper could have made and didn't.** It is
    the strongest constructive result we have. **Not affected by the §5 reframe.**
- **Also unwritten:** PAPER_2's slow register `s_i` ≈ FJ **stubbornness β_i**, but dynamic
  and self-generated. PAPER_2 is structurally *an FJ model with time-varying stubbornness*.

---

## §7. Files

```
/mnt/user-data/outputs/
  open_collective/            full codebase (README.md has the technical writeup)
  open_collective.zip
  SIMPLE_REPORT.md            plain-language version, no jargon, has a glossary
  HANDOFF.md                  this file
```

Run: `pip install numpy scipy matplotlib && python tests/test_theorems.py` → 10/10.

**User context:** knows the domain but not the standard parameter names — prefers plain
language and explicit definitions (see `SIMPLE_REPORT.md` for the register that worked).
Has repeatedly and correctly caught over-claims. **Treat pushback as signal.**

---

## §8. Key references

**Empirical (starlings)**
- Ballerini et al., *PNAS* **105**, 1232 (2008) — topological, 6–7 neighbours
- Ballerini et al., *Animal Behaviour* **76**, 201 (2008) — benchmark study, border effects
- Cavagna et al., *PNAS* **107**, 11865 (2010) — scale-free correlations
- Bialek et al., *PNAS* **109**, 4786 (2012) — maximum entropy for natural flocks
- **Pearce, Miller, Rowlands & Turner, *PNAS* **111**, 10422 (2014) — projection, marginal opacity ← THE ONE**
- Attanasi et al., *Nature Physics* **10**, 691 (2014) — linear undamped info transfer, inertial spin
- Young et al., *PLoS Comput Biol* **9**, e1002894 (2013) — why seven, under sensing uncertainty

**Theory**
- Vicsek et al., *PRL* **75**, 1226 (1995); Toner & Tu; Grégoire & Chaté
- Couzin et al., *J. Theor. Biol.* **218**, 1 (2002)
- D'Orsogna et al., *PRL* **96**, 104302 (2006)
- Cucker & Smale, *IEEE TAC* **52**, 852 (2007)
- Olfati-Saber, *IEEE TAC* **51**, 401 (2006)
- Ginelli & Chaté — metric-free models, zero-density steady state
- Altafini, *IEEE TAC* **58**, 935 (2013) — bipartite consensus, structural balance
- Shrinate & Tripathy, arXiv 2509.11038 — signed Friedkin–Johnsen
