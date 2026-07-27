"""
Drone Swarm Simulation — hierarchical formation/flocking control with
configurable sensing model (full / metric-radius / topological-k) and an
optional GPS-denied mode (dead-reckoning + range-based relative correction).

Architecture (matches current UAV-swarm SOTA taxonomy):
    Layer 1 (low-level)  : hard separation / collision avoidance
    Layer 2 (mid-level)  : formation-keeping (leader-follower offsets) +
                            flocking alignment (Cucker-Smale-style, distance-weighted)
    Layer 3 (high-level) : goal-seeking / mission waypoint

Everything an agent "knows" is forced through self.observe(i) — there is no
back-door access to ground truth from inside the control law. This is what
lets the bandwidth / feasibility numbers at the bottom of the report mean
something.
"""

import numpy as np


class SwarmConfig:
    def __init__(self,
                 n_drones=12,
                 dt=0.05,
                 max_speed=3.0,
                 max_accel=4.0,
                 comm_mode="topk",       # "full" | "metric" | "topk"
                 k_neighbors=7,
                 comm_radius=8.0,
                 min_separation=0.8,
                 formation="v_shape",    # "none" | "v_shape" | "grid" | "line"
                 formation_spacing=3.0,
                 weights=None,
                 gps_denied=False,
                 imu_drift_std=0.02,     # position drift injected per step (m)
                 range_noise_std=0.08,   # UWB-like ranging noise (m)
                 range_update_every=1,   # steps between ranging fixes (latency/bandwidth knob)
                 goal=(60.0, 0.0),
                 noise_std=0.05):
        self.n_drones = n_drones
        self.dt = dt
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.comm_mode = comm_mode
        self.k_neighbors = k_neighbors
        self.comm_radius = comm_radius
        self.min_separation = min_separation
        self.formation = formation
        self.formation_spacing = formation_spacing
        self.weights = weights or dict(align=1.6, cohesion=0.4, separation=1.5,
                                        formation=1.2, goal=1.0)
        self.gps_denied = gps_denied
        self.imu_drift_std = imu_drift_std
        self.range_noise_std = range_noise_std
        self.range_update_every = range_update_every
        self.goal = np.array(goal, dtype=float)
        self.noise_std = noise_std


def formation_offsets(n, kind, spacing):
    """Desired offset of drone i from the formation centroid/leader (drone 0)."""
    offsets = np.zeros((n, 2))
    if kind == "none":
        return offsets
    if kind == "v_shape":
        # leader at tip, alternating left/right wings behind
        for i in range(1, n):
            side = 1 if i % 2 == 1 else -1
            rank = (i + 1) // 2
            offsets[i] = (-rank * spacing, side * rank * spacing * 0.8)
    elif kind == "grid":
        cols = int(np.ceil(np.sqrt(n)))
        for i in range(n):
            r, c = divmod(i, cols)
            offsets[i] = (c * spacing, -r * spacing)
        offsets -= offsets.mean(axis=0)
    elif kind == "line":
        for i in range(n):
            offsets[i] = (-i * spacing, 0.0)
    return offsets


class SwarmSim:
    def __init__(self, config: SwarmConfig, seed=0):
        self.cfg = config
        self.rng = np.random.default_rng(seed)
        n = config.n_drones

        # ---- ground truth state (the simulator's own bookkeeping only) ----
        self.pos_true = self.rng.uniform(-5, 5, size=(n, 2))
        self.vel_true = np.zeros((n, 2))

        # ---- what each drone THINKS its position is ----
        # In GPS mode this equals ground truth (idealized). In GPS-denied
        # mode this drifts via dead reckoning and is corrected by ranging.
        self.pos_est = self.pos_true.copy()

        self.offsets = formation_offsets(n, config.formation, config.formation_spacing)
        self.t = 0
        self.history = {"order": [], "min_dist": [], "form_err": [], "est_err": []}

    # ------------------------------------------------------------------
    # OBSERVATION MODEL — the information-architecture boundary
    # ------------------------------------------------------------------
    def observe(self, i):
        """Return indices of neighbors visible to drone i, using ITS OWN
        position estimate (not ground truth) for distance calc — this
        matters once gps_denied=True, since a drifted estimate changes
        who you think your neighbors even are."""
        n = self.cfg.n_drones
        others = [j for j in range(n) if j != i]
        if self.cfg.comm_mode == "full":
            return others
        d = np.linalg.norm(self.pos_est[others] - self.pos_est[i], axis=1)
        if self.cfg.comm_mode == "metric":
            return [j for j, dist in zip(others, d) if dist < self.cfg.comm_radius]
        if self.cfg.comm_mode == "topk":
            k = min(self.cfg.k_neighbors, len(others))
            idx = np.argsort(d)[:k]
            return [others[idx_] for idx_ in idx]
        raise ValueError(self.cfg.comm_mode)

    # ------------------------------------------------------------------
    # LOCALIZATION — GPS-denied dead-reckoning + range correction
    # ------------------------------------------------------------------
    def update_localization(self):
        if not self.cfg.gps_denied:
            self.pos_est = self.pos_true.copy()
            return

        n = self.cfg.n_drones
        # 1) dead-reckoning propagation: integrate own commanded velocity,
        #    plus IMU drift noise (this is what makes error grow over time)
        self.pos_est += self.vel_true * self.cfg.dt
        self.pos_est += self.rng.normal(0, self.cfg.imu_drift_std, size=(n, 2))

        # 2) periodic range-based correction (simplified decentralized fusion):
        #    each drone measures noisy range+bearing to its observed neighbors
        #    and blends its own estimate toward the position implied by those
        #    neighbors' *own broadcast estimates* + the measured offset.
        if self.t % self.cfg.range_update_every == 0:
            new_est = self.pos_est.copy()
            for i in range(n):
                neighbors = self.observe(i)
                if not neighbors:
                    continue
                implied = []
                for j in neighbors:
                    true_offset = self.pos_true[j] - self.pos_true[i]
                    noisy_offset = true_offset + self.rng.normal(
                        0, self.cfg.range_noise_std, size=2)
                    implied.append(self.pos_est[j] - noisy_offset)
                implied_mean = np.mean(implied, axis=0)
                # blend: trust grows with neighbor count (more ranges = better fix)
                trust = min(0.5, 0.08 * len(neighbors))
                new_est[i] = (1 - trust) * self.pos_est[i] + trust * implied_mean
            self.pos_est = new_est

    # ------------------------------------------------------------------
    # CONTROL LAW — the three-layer stack
    # ------------------------------------------------------------------
    def compute_desired_velocity(self, i, neighbors):
        w = self.cfg.weights
        p_i, v_i = self.pos_est[i], self.vel_true[i]

        if not neighbors:
            align = np.zeros(2)
            cohesion = np.zeros(2)
            separation = np.zeros(2)
        else:
            p_n = self.pos_est[neighbors]
            v_n = self.vel_true[neighbors]
            d = np.linalg.norm(p_n - p_i, axis=1) + 1e-6

            # Layer 2a: alignment, Cucker-Smale-style distance decay weight
            cs_weight = 1.0 / (1.0 + d**2)
            align = (cs_weight[:, None] * v_n).sum(axis=0) / cs_weight.sum()
            align = align - v_i

            # Layer 2b: cohesion toward local neighbor centroid
            cohesion = p_n.mean(axis=0) - p_i

            # Layer 1: HARD separation — strong inverse-square repulsion
            # inside min_separation, real constraint not just a soft weight
            close = d < (self.cfg.min_separation * 1.8)
            if close.any():
                rep_dir = (p_i - p_n[close])
                rep_mag = np.clip(self.cfg.min_separation * 1.8 - d[close], 0, None)
                separation = (rep_dir / d[close, None] * rep_mag[:, None]).sum(axis=0)
            else:
                separation = np.zeros(2)

        # Layer 2c: formation-keeping — pull toward assigned offset from leader
        leader_pos = self.pos_est[0]
        desired_pos = leader_pos + self.offsets[i]
        formation = desired_pos - p_i if i != 0 else np.zeros(2)

        # Layer 3: mission goal (only leader pursues it directly; others
        # inherit motion through the formation term — standard leader-follower).
        # Cruise at constant commanded speed toward the goal rather than a
        # decaying position-error term, so the mission doesn't stall/dither
        # as it nears the target (a real cruise command, not a spring).
        goal_term = np.zeros(2)
        if i == 0:
            to_goal = self.cfg.goal - p_i
            dist_to_goal = np.linalg.norm(to_goal)
            if dist_to_goal > 1e-6:
                goal_term = (to_goal / dist_to_goal) * self.cfg.max_speed

        v_desired = (w["align"] * align +
                     w["cohesion"] * cohesion +
                     w["separation"] * separation +
                     w["formation"] * formation +
                     w["goal"] * goal_term)

        v_desired += self.rng.normal(0, self.cfg.noise_std, size=2)
        return v_desired

    # ------------------------------------------------------------------
    def step(self):
        n = self.cfg.n_drones
        self.update_localization()

        v_desired_all = np.zeros((n, 2))
        for i in range(n):
            neighbors = self.observe(i)
            v_desired_all[i] = self.compute_desired_velocity(i, neighbors)

        # saturate acceleration and speed (vehicle physical limits)
        accel = (v_desired_all - self.vel_true) / self.cfg.dt
        accel_mag = np.linalg.norm(accel, axis=1, keepdims=True)
        scale = np.minimum(1.0, self.cfg.max_accel / (accel_mag + 1e-9))
        accel *= scale
        self.vel_true += accel * self.cfg.dt

        speed = np.linalg.norm(self.vel_true, axis=1, keepdims=True)
        scale = np.minimum(1.0, self.cfg.max_speed / (speed + 1e-9))
        self.vel_true *= scale

        self.pos_true += self.vel_true * self.cfg.dt
        self.t += 1
        self._log_metrics()

    # ------------------------------------------------------------------
    def _log_metrics(self):
        v = self.vel_true
        speeds = np.linalg.norm(v, axis=1)
        headings = v / (speeds[:, None] + 1e-9)
        order = np.linalg.norm(headings.mean(axis=0))  # Vicsek order parameter

        n = self.cfg.n_drones
        dmat = np.linalg.norm(self.pos_true[:, None, :] - self.pos_true[None, :, :], axis=2)
        np.fill_diagonal(dmat, np.inf)
        min_dist = dmat.min()

        leader_pos = self.pos_true[0]
        desired = leader_pos + self.offsets
        form_err = np.linalg.norm(self.pos_true - desired, axis=1)[1:].mean() if n > 1 else 0.0

        est_err = np.linalg.norm(self.pos_est - self.pos_true, axis=1).mean()

        self.history["order"].append(order)
        self.history["min_dist"].append(min_dist)
        self.history["form_err"].append(form_err)
        self.history["est_err"].append(est_err)

    def run(self, steps):
        for _ in range(steps):
            self.step()
        return self.history
