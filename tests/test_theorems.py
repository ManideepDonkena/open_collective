"""
Validation suite. Each test checks a model against a PUBLISHED THEOREM, not against
a remembered number. Run: python -m pytest tests/ -v   (or just python tests/test_theorems.py)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from core import init_state, make_boundary, run
from core.metrics import (cluster_count, disagreement, fragmentation,
                          group_expansion, group_integrity, mixing_baseline,
                          polar_order, segregation_index)
from models.consensus import (AltafiniBipartite, DeGroot, FriedkinJohnsen,
                              SignedFJ, condensation_leaders, structural_balance)
from models.cohesive import CuckerSmaleModel, DOrsognaModel
from models.grouping import MultiGroupFlock, init_grouped_state
from models.formation import (CyclicPursuit, DisplacementFormation,
                              algebraic_connectivity, complete_graph,
                              cycle_graph, is_infinitesimally_rigid,
                              regular_polygon)


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    assert cond, msg


# ---------------------------------------------------------------- boundaries
def test_minimum_image():
    """Periodic displacement must use the minimum image; open must not."""
    per = make_boundary("periodic", L=10.0)
    opn = make_boundary("open")
    a, b = np.array([1.0, 1.0]), np.array([9.0, 9.0])
    check(np.allclose(per.displacement(a, b), [-2.0, -2.0]),
          "periodic uses minimum image (dist 2.83, not 11.3)")
    check(np.allclose(opn.displacement(a, b), [8.0, 8.0]),
          "open uses plain difference")
    check(per.distance(a, b) < opn.distance(a, b),
          "torus distance < free-space distance: THIS is the artifact")


# ---------------------------------------------------------------- Cucker-Smale
def test_cucker_smale_threshold():
    """THEOREM (Cucker & Smale 2007): beta <= 1/2 => unconditional flocking.

    Test: beta=0.4 must keep R_g bounded in OPEN space; beta=0.9 must not.
    """
    b = make_boundary("open")
    out = {}
    for beta in (0.4, 0.9):
        st = init_state(60, b, speed=0.5, spread=8.0, rng=np.random.default_rng(0))
        m = CuckerSmaleModel(b, K=2.0, sigma=1.0, beta=beta,
                             rng=np.random.default_rng(1))
        _, h = run(m, st, steps=800, dt=0.05, r_link=1.5, record_every=100)
        out[beta] = h["radius_of_gyration"][-1] / h["radius_of_gyration"][0]
    check(CuckerSmaleModel(b, beta=0.4).unconditional, "beta=0.4 flagged unconditional")
    check(not CuckerSmaleModel(b, beta=0.9).unconditional, "beta=0.9 flagged conditional")
    check(out[0.4] < out[0.9],
          f"beta=0.4 expands less than beta=0.9 ({out[0.4]:.2f}x vs {out[0.9]:.2f}x)")
    check(out[0.4] < 1.6, f"beta=0.4 keeps R_g bounded in free space ({out[0.4]:.2f}x)")


def test_dorsogna_hstability():
    """D'Orsogna: C*l^2 > 1 => H-stable, group stays a bounded blob in free space."""
    b = make_boundary("open")
    # --- catastrophic regime: C*l^2 < 1 -> collapse into a dense core ---------
    cat = DOrsognaModel(b, Ca=0.5, la=2.0, Cr=1.0, lr=0.5, beta=2.0)
    check(cat.hstability_ratio < 1.0,
          f"Ca=.5,la=2,Cr=1,lr=.5 => C*l^2={cat.hstability_ratio:.3f} < 1 => CATASTROPHIC")
    check(abs(cat.terminal_speed - np.sqrt(1.0 / 2.0)) < 1e-9, "terminal speed = sqrt(a/b)")
    st = init_state(60, b, speed=0.5, spread=8.0, rng=np.random.default_rng(0))
    _, h = run(cat, st, steps=700, dt=0.02, r_link=1.5, record_every=100)
    g = h["radius_of_gyration"][-1] / h["radius_of_gyration"][0]
    check(g < 1.0, f"catastrophic regime CONTRACTS to a dense core ({g:.2f}x), as predicted")
    check(h["n_fragments"][-1] == 1, "and stays a single connected group in FREE SPACE")

    # --- H-stable regime: C*l^2 > 1 -> the group DISPERSES -------------------
    # Counter-intuitive but correct, and worth stating plainly: D'Orsogna's
    # cohesive mills / flocks / rings live in the CATASTROPHIC regime. H-stability
    # means the energy is extensive and the group has a fixed equilibrium density,
    # so at finite N with self-propulsion it spreads out like a gas.
    # MORAL: the same model both survives and evaporates in open space depending
    # only on (Ca, la, Cr, lr). Cohesion is a PARAMETER-REGIME property, not a
    # property of the model's name.
    hst = DOrsognaModel(b, Ca=0.5, la=1.0, Cr=2.0, lr=0.8, beta=2.0, cutoff=10.0)
    check(hst.hstability_ratio > 1.0,
          f"Ca=.5,la=1,Cr=2,lr=.8 => C*l^2={hst.hstability_ratio:.3f} > 1 => H-STABLE")
    st2 = init_state(60, b, speed=0.5, spread=8.0, rng=np.random.default_rng(0))
    _, h2 = run(hst, st2, steps=1400, dt=0.02, r_link=2.5, record_every=50)
    g2 = h2["radius_of_gyration"][-1] / h2["radius_of_gyration"][0]
    check(g2 > 3.0,
          f"H-stable regime DISPERSES in free space ({g2:.2f}x) -- gas-like, not a flock")
    check(h2["n_fragments"][-1] > 10,
          f"=> and shatters into {h2['n_fragments'][-1]} fragments. Same model, "
          "opposite fate: cohesion is a regime, not a label.")


# ---------------------------------------------------------------- Altafini
def test_structural_balance_and_bipartite():
    """THEOREM (Altafini 2013): balanced+connected => bipartite consensus;
    unbalanced+connected => collapse to zero."""
    n = 10
    A = complete_graph(n).copy()
    A[:5, 5:] = -1.0
    A[5:, :5] = -1.0
    bal, part = structural_balance(A)
    check(bal, "two mutually hostile camps => structurally BALANCED")
    check(len(np.unique(part)) == 2, "gauge partition recovers exactly 2 camps")

    x0 = np.random.default_rng(0).uniform(-1, 1, n)
    xf, _ = AltafiniBipartite(A).run(x0, steps=6000, dt=0.005)
    check(cluster_count(xf, 0.05) == 2, "balanced => 2 opinion camps survive")
    check(np.abs(xf).mean() > 0.05, f"balanced => nonzero magnitude ({np.abs(xf).mean():.3f})")

    Au = A.copy()
    Au[0, 1] = Au[1, 0] = -1.0          # break the balance with one edge
    bal2, _ = structural_balance(Au)
    check(not bal2, "flipping ONE edge destroys structural balance")
    xf2, _ = AltafiniBipartite(Au).run(x0, steps=6000, dt=0.005)
    check(np.abs(xf2).mean() < 1e-2,
          f"unbalanced => collapse to zero ({np.abs(xf2).mean():.2e})")


def test_beuria_phi_minus_is_a_signed_network():
    """THE BRIDGE: PAPER_1's Phi- 'destructive interference' IS a signed network.

    PAPER_1 Eq. 32 gives p_i(t+1) = p_i + kappa[-(1/2)(p_i1+p_i2) + eta e_i].
    Every neighbour enters with a MINUS sign. As a signed graph that is the
    all-negative complete graph. Structural balance says: an all-negative complete
    graph on n>2 vertices is UNBALANCED (any triangle has 3 negative edges, an odd
    number). Altafini's theorem then predicts collapse to zero -- which is exactly
    the disorder Beuria reports numerically, but now with a REASON.
    """
    n = 6
    A_allneg = -(complete_graph(n))
    bal, _ = structural_balance(A_allneg)
    check(not bal, "all-negative complete graph (Phi- analogue) is UNBALANCED")
    x0 = np.random.default_rng(2).uniform(-1, 1, n)
    xf, _ = AltafiniBipartite(A_allneg).run(x0, steps=6000, dt=0.002)
    check(np.abs(xf).mean() < 1e-2,
          f"=> Altafini predicts collapse, matching PAPER_1's 'no order' ({np.abs(xf).mean():.2e})")

    # And the regime PAPER_1 never tested: a BALANCED signed perception state.
    A_bal = complete_graph(n).copy()
    A_bal[:3, 3:] = -1.0
    A_bal[3:, :3] = -1.0
    bal3, part3 = structural_balance(A_bal)
    xf3, _ = AltafiniBipartite(A_bal).run(x0, steps=6000, dt=0.005)
    check(bal3, "3-vs-3 signed perception state IS structurally balanced")
    signs = np.sign(xf3)
    check(len(np.unique(signs)) == 2, f"=> opinions split into 2 opposed camps {np.round(xf3,4)}")
    check(np.allclose(signs, part3) or np.allclose(signs, -part3),
          "=> the camps match the structural-balance gauge partition exactly")
    check(np.abs(xf3).std() < 1e-6,
          "=> the two camps are symmetric: +c and -c. This is BIPARTITE CONSENSUS,")
    print("        ^ this regime is absent from PAPER_1: a prediction it never made.")


# ---------------------------------------------------------------- consensus
def test_degroot_and_fj():
    """DeGroot converges iff primitive; FJ sustains disagreement via stubbornness."""
    n = 8
    W = complete_graph(n)
    x0 = np.random.default_rng(0).uniform(-1, 1, n)
    dg = DeGroot(W)
    check(dg.converges, "complete graph => primitive => DeGroot converges")
    xf, _ = dg.run(x0, 300)
    check(disagreement(xf) < 1e-9, f"DeGroot reaches consensus ({disagreement(xf):.1e})")
    check(abs(dg.social_power().sum() - 1.0) < 1e-9, "social power sums to 1")

    lam = np.full(n, 0.8)
    lam[[0, 1]] = 0.0                      # two fully stubborn agents
    fj = FriedkinJohnsen(W, lam, x0)
    xf2, _ = fj.run(steps=600)
    check(disagreement(xf2) > 1e-3,
          f"FJ with stubborn agents keeps disagreement alive ({disagreement(xf2):.3f})")
    eq = fj.equilibrium()
    check(np.allclose(eq, xf2, atol=1e-4), "FJ iterate matches closed-form equilibrium")


def test_condensation_leaders():
    """Opinion leaders = sinks of the condensation graph (Shrinate & Tripathy)."""
    A = np.zeros((5, 5))
    A[1, 0] = A[2, 0] = A[3, 1] = A[4, 2] = 1.0   # everyone flows toward 0
    _, _, leaders = condensation_leaders(A)
    check(list(leaders) == [0], f"agent 0 is the unique opinion leader (got {leaders})")


# ---------------------------------------------------------------- formation
def test_displacement_formation():
    """Converges to the target shape iff lambda_2 > 0 (connected)."""
    b = make_boundary("open")
    n = 8
    shape = regular_polygon(n, radius=3.0)
    A = complete_graph(n)
    check(algebraic_connectivity(A) > 0, "complete graph is connected (lambda_2 > 0)")
    st = init_state(n, b, spread=6.0, rng=np.random.default_rng(1))
    m = DisplacementFormation(b, A, shape, k=0.5)
    sf, _ = run(m, st, steps=800, dt=0.02, r_link=5.0, record_every=400)
    err = np.linalg.norm((sf.positions - sf.positions.mean(0))
                         - (shape - shape.mean(0)), axis=1).mean()
    check(err < 1e-3, f"formation converges to target shape in FREE SPACE (err={err:.2e})")

    A_disc = np.zeros((n, n))              # totally disconnected
    check(algebraic_connectivity(A_disc) < 1e-9, "disconnected graph => lambda_2 = 0")


def test_rigidity_gate():
    """Distance-based formation needs infinitesimal rigidity: rank(R) = 2n-3."""
    shape = regular_polygon(5, radius=2.0)
    check(is_infinitesimally_rigid(shape, complete_graph(5)),
          "complete graph on 5 pts is infinitesimally rigid")
    check(not is_infinitesimally_rigid(shape, cycle_graph(5)),
          "5-cycle is NOT rigid (it flexes) => distance control will not converge")


def test_cyclic_pursuit_polygon():
    """alpha = pi/n gives a stable regular n-gon (Marshall/Broucke/Francis;
    Tripathy & Shima, Automatica 2024)."""
    b = make_boundary("open")
    n = 8
    st = init_state(n, b, spread=5.0, rng=np.random.default_rng(7))
    m = CyclicPursuit(b, alpha=CyclicPursuit.polygon_alpha(n), k=0.5)
    sf, _ = run(m, st, steps=2000, dt=0.02, r_link=10.0, record_every=1000)
    P = np.vstack([sf.positions, sf.positions[:1]])
    side = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cv = side.std() / side.mean()
    check(cv < 0.02, f"alpha=pi/{n} converges to a regular {n}-gon (side CV={cv:.4f})")

    st2 = init_state(n, b, spread=5.0, rng=np.random.default_rng(7))
    m2 = CyclicPursuit(b, alpha=0.0, k=0.5)
    sf2, _ = run(m2, st2, steps=2000, dt=0.02, r_link=10.0, record_every=1000)
    spread = np.linalg.norm(sf2.positions - sf2.positions.mean(0), axis=1).mean()
    check(spread < 0.05, f"alpha=0 => rendezvous at the centroid (spread={spread:.2e})")


# ---------------------------------------------------------------- grouping
def test_fusion_is_invisible_to_cohesion_metrics():
    """The claim Experiment 2 exists to make: cohesion metrics CANNOT see fusion.

    Delete the segregation term and the groups merge. Every Experiment-1 metric
    (largest_cluster_frac, n_fragments) reports a PERFECT score for that run, while
    the segregation index collapses. If a single cohesion number could detect it,
    core/metrics.py's group section would not need to exist.
    """
    b = make_boundary("open", dim=3)
    out = {}
    for w_seg in (2.5, 0.0):
        st = init_grouped_state(90, 3, b, dim=3, blob_radius=0.9, arena=8.0,
                                rng=np.random.default_rng(0))
        m = MultiGroupFlock(b, beta=0.4, w_seg=w_seg, w_global=0.05,
                            rng=np.random.default_rng(1))
        sf, _ = run(m, st, steps=1500, dt=0.05, r_link=1.5, record_every=500)
        g = sf.internal["groups"]
        out[w_seg] = dict(
            S=segregation_index(sf.positions, g, b),
            largest=fragmentation(sf.positions, b, 1.5)[1],
            integrity=group_integrity(sf.positions, g, b, 1.5),
        )
    check(out[2.5]["S"] > 0.75, f"segregation on  => groups distinct (S={out[2.5]['S']:.2f})")
    check(out[0.0]["S"] < 0.60, f"segregation off => groups FUSE (S={out[0.0]['S']:.2f})")
    check(out[0.0]["largest"] > 0.95,
          f"...yet largest_cluster_frac reports a PERFECT {out[0.0]['largest']:.2f} "
          f"for the fused run: cohesion metrics are blind to fusion")
    check(out[0.0]["integrity"] > 0.95,
          f"...and group_integrity is {out[0.0]['integrity']:.2f} too: the groups "
          f"lost no members, only their identity")


def test_grouping_beta_threshold_is_unconditional():
    """CS beta<=1/2 is UNCONDITIONAL: no expansion from ANY initial condition.

    Per group, with no arena term, so only the in-group weight holds a group. The
    signature of 'unconditional' is not merely that beta=0.4 is better -- it is that
    beta=0.4 is insensitive to the initial condition while beta=0.9 is not.
    """
    b = make_boundary("open", dim=3)
    g = {}
    for beta in (0.4, 0.9):
        for blob in (0.9, 3.5):
            st = init_grouped_state(90, 3, b, dim=3, blob_radius=blob, arena=24.0,
                                    rng=np.random.default_rng(0))
            m = MultiGroupFlock(b, beta=beta, w_seg=2.5, w_global=0.0,
                                rng=np.random.default_rng(1))
            Rg0 = group_expansion(st.positions, st.internal["groups"], b)
            sf, _ = run(m, st, steps=3000, dt=0.05, r_link=1.5, record_every=1000)
            g[(beta, blob)] = group_expansion(sf.positions, sf.internal["groups"],
                                              b) / Rg0
    for blob in (0.9, 3.5):
        check(g[(0.4, blob)] < g[(0.9, blob)],
              f"blob={blob}: beta=0.4 expands less than beta=0.9 "
              f"({g[(0.4, blob)]:.2f}x vs {g[(0.9, blob)]:.2f}x)")
    check(g[(0.4, 0.9)] < 1.25 and g[(0.4, 3.5)] < 1.25,
          f"beta=0.4 barely expands from EITHER start "
          f"({g[(0.4, 0.9)]:.2f}x tight, {g[(0.4, 3.5)]:.2f}x spread) = unconditional")
    check(g[(0.9, 3.5)] > 1.25,
          f"beta=0.9 expands from the spread start ({g[(0.9, 3.5)]:.2f}x): its "
          f"flocking is conditional, so the initial condition is allowed to matter")


def test_group_metrics_baseline():
    """segregation_index must be gauge-correct: 0 when mixed, 1 when segregated.

    Raw purity is not interpretable -- with 2 equal groups a totally fused cloud
    still scores ~0.5. The baseline subtraction is what makes the number mean
    something, so it is worth testing directly rather than trusting.
    """
    b = make_boundary("open", dim=3)
    rng = np.random.default_rng(0)
    # perfectly segregated: two blobs 50 units apart
    x = np.vstack([rng.normal(0, 1, (40, 3)), rng.normal(50, 1, (40, 3))])
    g = np.repeat([0, 1], 40)
    check(segregation_index(x, g, b) > 0.95,
          f"two separated blobs => S ~ 1 ({segregation_index(x, g, b):.3f})")
    # well mixed: same cloud, labels assigned at random
    x = rng.normal(0, 1, (80, 3))
    g = rng.permutation(np.repeat([0, 1], 40))
    S = segregation_index(x, g, b)
    check(abs(S) < 0.25, f"randomly mixed labels => S ~ 0 ({S:.3f})")
    check(abs(mixing_baseline(np.repeat([0, 1], 40)) - 0.4937) < 1e-3,
          "mixing_baseline(2 equal groups of 40) = 39/79 = 0.494, not 0.5")


ALL = [test_minimum_image, test_cucker_smale_threshold, test_dorsogna_hstability,
       test_structural_balance_and_bipartite, test_beuria_phi_minus_is_a_signed_network,
       test_degroot_and_fj, test_condensation_leaders, test_displacement_formation,
       test_rigidity_gate, test_cyclic_pursuit_polygon,
       test_fusion_is_invisible_to_cohesion_metrics,
       test_grouping_beta_threshold_is_unconditional,
       test_group_metrics_baseline]

if __name__ == "__main__":
    fails = 0
    for t in ALL:
        print(f"\n{t.__name__}\n  {t.__doc__.strip().splitlines()[0]}")
        try:
            t()
        except AssertionError as e:
            fails += 1
            print(f"  !! {e}")
    print("\n" + "=" * 70)
    print(f"{len(ALL) - fails}/{len(ALL)} theorem tests passed")
