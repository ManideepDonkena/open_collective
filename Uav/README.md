# Drone Swarm Formation Sim — Results & Hardware Mapping

> ### ⚠️ Read this first: three findings from the flocking side
>
> `multi_team_3d.py` + `run_multi_team.py` re-run this controller as **K teams in 3D open
> airspace**, with turbulence and a **leader-loss** event. Full write-up:
> **[../RESEARCH.md](../RESEARCH.md) §5**. The short version:
>
> 1. **`cs_weight = 1.0/(1.0 + d**2)` is not "Cucker–Smale-style".** It is Cucker–Smale's
>    ψ at **β = 1**, outside the **β ≤ 1/2** regime the theorem requires — and dividing by
>    `cs_weight.sum()` makes it a weighted *average* (a DeGroot/Vicsek consensus step), which
>    the theorem does not describe at **any** β. The comment is wrong either way; someone
>    will eventually trust it.
> 2. **It works anyway — and the normalisation is why.** A lone straggler's only neighbours
>    are its own team, so an average gives them full weight at any range and its velocity
>    re-converges (gap bounded at 22/63/123 m). Sound, but by luck, not by theorem.
>    **The trap is the obvious fix:** unnormalise into a "real" CS sum while leaving β=1 and
>    the drone is **lost** — gap 548/1202/1408 m and still growing; the team expands **31×**
>    under turbulence. *Half-adopting the theory is far worse than ignoring it.* To get the
>    guarantee, unnormalise **and** set β≈0.4–0.5. Cost: one exponent.
> 3. **The leader is a single point of failure for cohesion, and the order parameter cannot
>    warn you.** While the leader lives, the swarm is cohesive because of the *formation*
>    term, for reasons having nothing to do with the flocking layer. Test leader loss.
>
> Two further cautions this repo learned the hard way:
> **run length is load-bearing** — at t=120 s every rule scored an identical, perfect 1.01×
> and the table read "all safe"; the β=1 leak only appeared by t=400 s, because dispersal is
> a *rate*. And **no alignment rule recovers a straggler**: 22 m at t=100 s is 22 m at
> t=600 s. CS gives velocity consensus and a *bounded* gap; it never promised to close one.
> If your CONOPS assumes a displaced drone rejoins its slot unaided, no β delivers that.

## What this simulates
A 12-drone V-formation flying a cruise mission, under 4 scenarios:

| Scenario | Comm model | GPS | What it tests |
|---|---|---|---|
| A | Full (all-to-all) | Yes | Idealized upper bound — not realistic bandwidth |
| B | Topological k=7 | Yes | Realistic bandwidth, still has GPS |
| C | Topological k=7 | **No** — UWB ranging + dead-reckoning, corrected every step | Your actual scenario |
| D | Topological k=7 | No — ranging only every 5 steps | Lower-bandwidth / more realistic radio duty cycle |

## Results (after tuning out two bugs — see "what broke" below)

- **Order parameter** (group heading alignment, 1.0 = perfect): all four scenarios converge to
  **0.95–1.0** after the initial transient. Topological-k and GPS-denied cost you almost nothing
  in coherence once the controller is tuned correctly.
- **Formation error**: full-comm scenario A is actually *worse* (2.48 m) than topological
  scenario B (1.69 m) — because with all-to-all cohesion, distant "neighbors" pull agents
  off their assigned V-slot. **This is the practical case for k-nearest sensing being not just
  cheaper but better-behaved**, not only a bandwidth compromise.
- **Localization error (GPS-denied)**: settles around **0.15–0.21 m** mean, driven upward mostly
  by IMU drift between ranging fixes. Sparser ranging (scenario D, every 5 steps) roughly halves
  the correction rate but only costs ~0.05 m extra error — meaning you likely have headroom to
  cut ranging-radio duty cycle substantially before it hurts formation quality.
- **Minimum separation** never dropped below 2.1 m in any scenario (safety floor set at 0.8 m) —
  the hard-avoidance layer held under all four conditions, including the noisiest one.

## What broke during tuning (worth knowing before you touch hardware)

1. **Separation "safety bubble" overlapped the intended formation spacing** — drones assigned to
   fly 2 m apart in the V were also being told "you're too close, push away" by the collision
   layer, so they fought each other and the whole formation oscillated. Fix: make sure your
   collision-avoidance trigger radius is *smaller* than your tightest intentional formation spacing,
   with margin. This is a real failure mode people hit on actual hardware, not just in sim.
2. **Goal-seeking modeled as a decaying spring instead of a cruise command** — as the leader
   approached the waypoint, its own commanded speed decayed toward zero, so its heading became
   noise-dominated and the whole swarm's order parameter got noisy near the end of the run.
   Fix: command constant cruise speed toward the next waypoint, don't let it decay — matches how
   real flight controllers (PX4/ArduPilot offboard mode) expect velocity setpoints anyway.

## Mapping to real hardware

The `compute_desired_velocity()` output (`v_desired`, a 2D vector per drone per tick) is exactly
what you'd send as an **offboard velocity setpoint**:

- **PX4**: `MAVLink SET_POSITION_TARGET_LOCAL_NED` with velocity fields set, or via **MAVSDK**
  (`drone.offboard.set_velocity_ned(...)`), streamed at ≥2 Hz (PX4 requires a minimum setpoint
  rate in offboard mode or it will fail back to a safe mode).
- **ArduPilot**: same MAVLink message, `COPTER` offboard/guided mode.
- **Ranging**: replace the simulated range-noise block with real UWB module readings
  (e.g., Decawave/Qorvo DW1000/DW3000 modules give raw time-of-flight ranges — feed those into
  the same fusion step instead of `rng.normal(...)`).
- **Neighbor discovery for `observe()`**: your radio's broadcast/beacon layer determines who's
  "in range" — the topological-k logic here assumes you already have ranges to more candidates
  than k and are choosing the nearest k; if your radio itself is range-limited, use `"metric"`
  mode instead so it matches your radio's real reach.
- **dt**: sim uses dt=0.05 (20 Hz control loop) — match this to your actual companion-computer
  control loop rate (Pixhawk-class autopilots commonly run offboard velocity loops at 10–50 Hz).

## Tunable parameters (in `SwarmConfig`)

| Param | What it controls | Tuning direction |
|---|---|---|
| `weights.separation` | How hard drones push apart when too close | Increase if you see near-misses; decrease if formation can't tighten |
| `weights.formation` | How strongly followers snap to their V-slot | Increase for tighter formation-keeping, but watch for oscillation vs. separation |
| `weights.align` | How strongly followers match neighbor velocity | Higher = smoother group motion, damps oscillation |
| `k_neighbors` | Bandwidth/compute per drone | Lower = cheaper, but degrades formation precision if too low (try k=4–5 as a lower bound test) |
| `range_update_every` | Ranging radio duty cycle | Higher = lower power/bandwidth, more localization drift — scenario D shows the tradeoff is mild |
| `imu_drift_std` | Quality of your actual IMU | Set from your real hardware's datasheet, not a guess |
| `min_separation` | Hard safety floor | Set from your drone's physical prop-diameter + margin, non-negotiable |

## Next steps to make this production-real

1. Swap the flat 2D kinematic model for your drone's actual dynamics (quadrotor with thrust/attitude
   limits, not a velocity-controlled point mass) — this sim is a *behavioral* layer prototype, not
   a flight-dynamics validator. **Still open.**
2. Replace the simplified ranging-fusion blend with a proper distributed EKF (state = [x,y,vx,vy]
   per drone, standard cooperative-localization formulation) before trusting numbers past ~15 drones.
   **Still open** — and note `multi_team_3d.py` does *not* carry the localization layer over at all,
   so its numbers assume perfect state knowledge. The two models are complementary, not a merge.
3. ~~Add wind/battery-drain disturbance to `step()`~~ — **done in `multi_team_3d.py`**
   (`turbulence=`), and it was not cosmetic: with turbulence off, *every* alignment rule scores an
   identical, perfect R_g growth of 1.00 and the comparison says nothing. Note it must be
   **per-drone uncorrelated** — a uniform wind field displaces a whole team equally and tests
   cohesion not at all. Battery drain still open.
4. ~~Move from 2D to 3D~~ — **done in `multi_team_3d.py`**: 3D V-slots with a vertical stagger
   (`dihedral`), which is what buys altitude separation instead of leaning on lateral spacing.
   Visualize with `python Uav/run_multi_team.py --viz` (see `../core/viz3d.py`).
5. **New, and the one worth doing first: fix the alignment weight** — see the box at the top.
   It is a one-line change with a theorem attached, and the *wrong* one-line change loses drones.
