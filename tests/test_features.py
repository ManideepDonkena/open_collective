"""
Verify the optional-dependency / performance features:
  * VicsekModel(fast=True) matches the default loop and is a vectorised speed-up
  * experiments.manager pandas helpers (skips cleanly if pandas is absent)

    python tests/test_features.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from core import make_boundary
import core.init as cinit
from models import VicsekModel


def check_fast_vicsek():
    print("--- Vicsek fast path ---")
    b = make_boundary("periodic", L=10.0, dim=2)
    st = cinit.random_init(400, b, speed=0.5, rng=np.random.default_rng(1))
    slow = VicsekModel(b, r_max=1.0, eta=0.2, rng=np.random.default_rng(7))
    fast = VicsekModel(b, r_max=1.0, eta=0.2, fast=True, rng=np.random.default_rng(7))
    a = slow.step(st.copy(), 0.05)
    c = fast.step(st.copy(), 0.05)
    diff = np.abs(a.positions - c.positions).max()
    assert np.allclose(a.positions, c.positions, atol=1e-9), \
        f"fast path diverged in one step (max diff {diff:.1e})"
    print(f"  PASS  fast == slow after one step (max diff {diff:.1e})")

    s = c
    for _ in range(30):
        s = fast.step(s, 0.05)
    assert np.all(np.isfinite(s.positions))
    print("  PASS  fast path stays finite over 30 steps")

    big = cinit.random_init(1500, b, speed=0.5, rng=np.random.default_rng(2))
    ms = VicsekModel(b, rng=np.random.default_rng(0))
    mf = VicsekModel(b, fast=True, rng=np.random.default_rng(0))
    def timeit(model):
        s = big.copy()
        t0 = time.perf_counter()
        for _ in range(10):
            s = model.step(s, 0.05)
        return (time.perf_counter() - t0) * 1000
    tslow, tfast = timeit(ms), timeit(mf)
    print(f"  INFO  N=1500 x10 steps: loop {tslow:.0f} ms, fast {tfast:.0f} ms "
          f"({tslow / max(tfast, 1e-6):.1f}x)")


def check_pandas():
    print("\n--- pandas helpers ---")
    try:
        import pandas  # noqa: F401
    except Exception as e:
        print(f"  SKIP  pandas not installed ({e})")
        return
    from experiments import manager
    cfg = {
        "boundary": {"kind": "periodic", "L": 10.0, "dim": 2},
        "init": {"method": "random", "n": 80, "speed": 0.5},
        "model": {"name": "Vicsek", "params": {"r_max": 1.0, "eta": 0.2}},
        "run": {"steps": 60, "dt": 0.05, "r_link": 1.0, "record_every": 5},
    }
    final, hist = manager.run_experiment(cfg)
    df = manager.to_dataframe(hist)
    assert "t" in df.columns and "polar_order" in df.columns and len(df) > 0
    print(f"  PASS  to_dataframe -> {df.shape[0]} rows x {df.shape[1]} cols")


if __name__ == "__main__":
    print("=" * 50)
    check_fast_vicsek()
    check_pandas()
    print("\nfeatures OK.")
