from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from swarm_sim import SwarmConfig, SwarmSim

STEPS = 600

scenarios = {
    "A: Full comm, GPS available (idealized)": SwarmConfig(
        n_drones=12, comm_mode="full", gps_denied=False, formation="v_shape"),
    "B: Topological k=7, GPS available": SwarmConfig(
        n_drones=12, comm_mode="topk", k_neighbors=7, gps_denied=False, formation="v_shape"),
    "C: Topological k=7, GPS-DENIED (UWB ranging + dead-reckoning)": SwarmConfig(
        n_drones=12, comm_mode="topk", k_neighbors=7, gps_denied=True, formation="v_shape",
        imu_drift_std=0.02, range_noise_std=0.08, range_update_every=1),
    "D: Same as C but sparse ranging (every 5 steps = lower bandwidth)": SwarmConfig(
        n_drones=12, comm_mode="topk", k_neighbors=7, gps_denied=True, formation="v_shape",
        imu_drift_std=0.02, range_noise_std=0.08, range_update_every=5),
}

results = {}
for name, cfg in scenarios.items():
    sim = SwarmSim(cfg, seed=42)
    hist = sim.run(STEPS)
    results[name] = (sim, hist)
    print(f"{name}")
    print(f"   final order param      : {hist['order'][-1]:.3f}")
    print(f"   mean min-separation     : {np.mean(hist['min_dist']):.3f} m "
          f"(limit={cfg.min_separation} m)")
    print(f"   mean formation error    : {np.mean(hist['form_err'][-200:]):.3f} m (last 200 steps)")
    print(f"   mean localization error : {np.mean(hist['est_err']):.3f} m")
    print()

# ---------------------------------------------------------------
# Plot 1: metrics over time, all scenarios
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
t = np.arange(STEPS) * 0.05

for name, (sim, hist) in results.items():
    label = name.split(":")[0]
    axes[0, 0].plot(t, hist["order"], label=label)
    axes[0, 1].plot(t, hist["min_dist"], label=label)
    axes[1, 0].plot(t, hist["form_err"], label=label)
    axes[1, 1].plot(t, hist["est_err"], label=label)

axes[0, 0].set_title("Order parameter (1 = perfectly aligned heading)")
axes[0, 0].set_xlabel("time (s)"); axes[0, 0].set_ylim(0, 1.05); axes[0, 0].legend(fontsize=8)

axes[0, 1].axhline(1.2, color="red", ls="--", lw=1, label="min separation limit")
axes[0, 1].set_title("Minimum inter-drone distance (collision check)")
axes[0, 1].set_xlabel("time (s)"); axes[0, 1].legend(fontsize=8)

axes[1, 0].set_title("Formation error (mean distance from assigned V-slot)")
axes[1, 0].set_xlabel("time (s)"); axes[1, 0].legend(fontsize=8)

axes[1, 1].set_title("Localization error (|estimated pos - true pos|)")
axes[1, 1].set_xlabel("time (s)"); axes[1, 1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(str(Path(__file__).resolve().parent / "metrics_comparison.png"), dpi=140)
plt.close()

# ---------------------------------------------------------------
# Plot 2: final formation snapshot + trajectories for scenario C (the
# realistic GPS-denied case, since that's the scenario in question)
# ---------------------------------------------------------------
sim_c, hist_c = results["C: Topological k=7, GPS-DENIED (UWB ranging + dead-reckoning)"]

# re-run scenario C but log full trajectory this time
cfg_c = scenarios["C: Topological k=7, GPS-DENIED (UWB ranging + dead-reckoning)"]
sim2 = SwarmSim(cfg_c, seed=42)
traj = np.zeros((STEPS, cfg_c.n_drones, 2))
for s in range(STEPS):
    sim2.step()
    traj[s] = sim2.pos_true

fig, ax = plt.subplots(figsize=(9, 7))
for i in range(cfg_c.n_drones):
    ax.plot(traj[:, i, 0], traj[:, i, 1], lw=0.7, alpha=0.6)
ax.scatter(traj[-1, :, 0], traj[-1, :, 1], c="black", s=30, zorder=5, label="final positions")
ax.scatter(traj[0, :, 0], traj[0, :, 1], c="green", s=30, zorder=5, label="start positions")
ax.set_title("Scenario C — GPS-denied, topological k=7\nTrajectories and final V-formation")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.legend()
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(str(Path(__file__).resolve().parent / "formation_snapshot.png"), dpi=140)
plt.close()

print("Saved: metrics_comparison.png, formation_snapshot.png")
