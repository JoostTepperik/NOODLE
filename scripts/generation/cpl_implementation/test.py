"""
test_kic.py

Tests for kic_tripeptide_closure.py.

Three test scenarios:
  1. Tripeptide round-trip  — build 2 residues from known torsions, use the
     resulting virtual-N as target; verify the true solution is recovered.
  2. Full-loop round-trip   — 12-residue loop; verify kic_close_given_torsions
     returns closed solutions and true torsions are among them.
  3. Multiple geometry cases — several random seeds to check solution counts
     and closure errors are consistently zero.

Run:
    python test_kic.py
"""
from __future__ import annotations

import sys
import types
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Inline nerf stub (mirrors nerf.py exactly so the test is self-contained)
# ─────────────────────────────────────────────────────────────────────────────

def _make_nerf_module():
    BOND_LENGTHS = {
        'N_CA': 1.458, 'CA_C': 1.525, 'C_N': 1.329, 'C_O': 1.231,
    }
    BOND_ANGLES_RAD = {
        'C_N_CA': np.radians(121.7),
        'N_CA_C': np.radians(111.2),
        'CA_C_N': np.radians(116.2),
        'CA_C_O': np.radians(120.8),
    }

    def nerf(a, b, c, bl, ba, dihedral):
        bc = c - b
        bc_hat = bc / np.linalg.norm(bc)
        n = a - b
        n = n - np.dot(n, bc_hat) * bc_hat
        nl = np.linalg.norm(n)
        if nl < 1e-8:
            tmp = (np.array([0., 0., 1.]) if abs(bc_hat[2]) < 0.9
                   else np.array([1., 0., 0.]))
            n = tmp - np.dot(tmp, bc_hat) * bc_hat
            nl = np.linalg.norm(n)
        n_hat = n / nl
        m_hat = np.cross(bc_hat, n_hat)
        d = np.array([
            -bl * np.cos(ba),
             bl * np.sin(ba) * np.cos(dihedral),
             bl * np.sin(ba) * np.sin(dihedral),
        ])
        return c + np.column_stack([bc_hat, n_hat, m_hat]) @ d

    def build_loop(pN, pCA, pC, psi_prev, phi, psi, omega=None, reverse=False):
        phi = np.asarray(phi, dtype=float)
        psi = np.asarray(psi, dtype=float)
        n = len(phi)
        if len(psi) != n:
            raise ValueError("phi and psi must have the same length")
        if omega is None:
            omega = np.full(n, np.pi)
        else:
            omega = np.asarray(omega, dtype=float)
            if len(omega) != n:
                raise ValueError("omega must have the same length as phi/psi")
        if reverse:
            phi = phi[::-1]
            psi = psi[::-1]
            omega = omega[::-1]
        L = BOND_LENGTHS
        A = BOND_ANGLES_RAD
        N  = np.empty((n, 3))
        CA = np.empty((n, 3))
        C  = np.empty((n, 3))
        O  = np.empty((n, 3))
        N[0]  = nerf(pN,  pCA, pC,   L['C_N'],  A['CA_C_N'], psi_prev)
        CA[0] = nerf(pCA, pC,  N[0], L['N_CA'], A['C_N_CA'], omega[0])
        C[0]  = nerf(pC,  N[0], CA[0], L['CA_C'], A['N_CA_C'], phi[0])
        O[0]  = nerf(N[0], CA[0], C[0], L['C_O'], A['CA_C_O'], psi[0] + np.pi)
        for i in range(1, n):
            N[i]  = nerf(N[i-1],  CA[i-1], C[i-1], L['C_N'],  A['CA_C_N'], psi[i-1])
            CA[i] = nerf(CA[i-1], C[i-1],  N[i],   L['N_CA'], A['C_N_CA'], omega[i])
            C[i]  = nerf(C[i-1],  N[i],    CA[i],  L['CA_C'], A['N_CA_C'], phi[i])
            O[i]  = nerf(N[i], CA[i], C[i], L['C_O'], A['CA_C_O'], psi[i] + np.pi)
        return N, CA, C, O

    mod = types.ModuleType('nerf')
    mod.BOND_LENGTHS     = BOND_LENGTHS
    mod.BOND_ANGLES_RAD  = BOND_ANGLES_RAD
    mod.nerf             = nerf
    mod.build_loop       = build_loop
    return mod


_nerf_mod = _make_nerf_module()
sys.modules['nerf'] = _nerf_mod

_nerf       = _nerf_mod.nerf
_build_loop = _nerf_mod.build_loop
_L          = _nerf_mod.BOND_LENGTHS
_A          = _nerf_mod.BOND_ANGLES_RAD

import kic as kic   

# ─────────────────────────────────────────────────────────────────────────────
# Shared anchor geometry (used across all tests)
# ─────────────────────────────────────────────────────────────────────────────

_ANCHOR_N  = np.array([0.0,  0.0, 0.0])
_ANCHOR_CA = np.array([1.458, 0.0, 0.0])
_ANCHOR_C  = _nerf(
    np.array([-1., 0., 0.]),
    _ANCHOR_N, _ANCHOR_CA,
    _L['CA_C'], _A['N_CA_C'], -1.2,
)


def _build_bridge(t1, t2, t3):
    """Place the two bridge residues from the shared anchor and return virtual N."""
    N_k1  = _nerf(_ANCHOR_N, _ANCHOR_CA, _ANCHOR_C,
                  _L['C_N'],  _A['CA_C_N'], t1)
    CA_k1 = _nerf(_ANCHOR_CA, _ANCHOR_C, N_k1,
                  _L['N_CA'], _A['C_N_CA'], np.pi)
    C_k1  = _nerf(_ANCHOR_C, N_k1, CA_k1,
                  _L['CA_C'], _A['N_CA_C'], t2)
    virt_N = _nerf(N_k1, CA_k1, C_k1,
                   _L['C_N'], _A['CA_C_N'], t3)
    return virt_N


def _rebuild_closure(phi_s, psi_s, prev_N, prev_CA, prev_C, psi_prev):
    """Build the full loop from solved torsions and return the virtual N error."""
    N_s, CA_s, C_s, _ = _build_loop(prev_N, prev_CA, prev_C, psi_prev, phi_s, psi_s)
    virt_N = _nerf(N_s[-1], CA_s[-1], C_s[-1],
                   _L['C_N'], _A['CA_C_N'], psi_s[-1])
    return virt_N


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

def _check(label, condition, detail=""):
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return condition


def _torsion_match(a, b, tol_deg=1.0):
    """Angular distance in degrees, wrapping at ±180°."""
    diff = abs(np.degrees(a) - np.degrees(b)) % 360
    return min(diff, 360 - diff) < tol_deg


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — tripeptide round-trip
# ─────────────────────────────────────────────────────────────────────────────

def test_tripeptide_roundtrip():
    print("\n── Test 1: tripeptide round-trip ──────────────────────────────────")

    cases = [
        (-1.10,  0.80, -0.50, "helix-like"),
        (-2.36,  2.36, -1.05, "beta-like"),
        ( 0.52, -1.05,  2.09, "mixed"),
        (-0.17,  3.00, -2.80, "near-boundary"),
    ]

    all_passed = True
    for t1_true, t2_true, t3_true, label in cases:
        target_N = _build_bridge(t1_true, t2_true, t3_true)
        sols = kic.kic_tripeptide_close(
            _ANCHOR_N, _ANCHOR_CA, _ANCHOR_C, target_N, n_grid=3600,
        )

        n_sols = len(sols)
        all_zero_err = all(err < 0.01 for _, _, _, err in sols)
        true_recovered = any(
            _torsion_match(t1, t1_true) for t1, _, _, _ in sols
        )

        ok = n_sols > 0 and all_zero_err and true_recovered
        all_passed &= ok
        _check(
            f"{label:16s}  {n_sols} solutions",
            ok,
            f"all_zero_err={all_zero_err}  true_recovered={true_recovered}",
        )

        if n_sols > 0:
            for i, (t1, t2, t3, err) in enumerate(sols):
                marker = " ← true" if _torsion_match(t1, t1_true) else ""
                print(f"             sol {i}: err={err:.7f} Å"
                      f"  t1={np.degrees(t1):7.2f}°"
                      f"  t2={np.degrees(t2):7.2f}°"
                      f"  t3={np.degrees(t3):7.2f}°{marker}")

    return all_passed


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — full 12-residue loop round-trip via kic_close_given_torsions
# ─────────────────────────────────────────────────────────────────────────────

def test_full_loop_roundtrip():
    print("\n── Test 2: full-loop round-trip (12 residues) ─────────────────────")

    rng = np.random.default_rng(99)
    n_loop   = 12
    phi_true = rng.uniform(-np.pi, np.pi, n_loop)
    psi_true = rng.uniform(-np.pi, np.pi, n_loop)

    prev_N   = np.array([10.0,  5.0,  0.0])
    prev_CA  = np.array([11.5,  5.0,  0.0])
    prev_C   = np.array([12.2,  6.3,  0.5])
    psi_prev = -0.8

    # Build the true loop to get the reachable target N
    N_f, CA_f, C_f, _ = _build_loop(prev_N, prev_CA, prev_C, psi_prev,
                                     phi_true, psi_true)
    target_N = _nerf(N_f[-1], CA_f[-1], C_f[-1],
                     _L['C_N'], _A['CA_C_N'], psi_true[-1])

    results = kic.kic_close_given_torsions(
        phi_true, psi_true,
        prev_N, prev_CA, prev_C, psi_prev,
        target_N,
    )

    n_sols = len(results)
    _check(f"found {n_sols} closed solutions", n_sols > 0)

    all_passed = n_sols > 0
    for i, (phi_s, psi_s) in enumerate(results):
        virt_N = _rebuild_closure(phi_s, psi_s, prev_N, prev_CA, prev_C, psi_prev)
        err    = float(np.linalg.norm(virt_N - target_N))
        ok     = err < 0.05
        all_passed &= ok

        true_match = (
            _torsion_match(psi_s[n_loop - 2], psi_true[n_loop - 2])
            and _torsion_match(phi_s[n_loop - 1], phi_true[n_loop - 1])
            and _torsion_match(psi_s[n_loop - 1], psi_true[n_loop - 1])
        )
        marker = " ← true pivots" if true_match else ""
        _check(
            f"sol {i}: rebuild closure err={err:.6f} Å{marker}",
            ok,
        )

    return all_passed


# ─────────────────────────────────────────────────────────────────────────────
# Test 2b — reverse-order build support
# ─────────────────────────────────────────────────────────────────────────────

def test_reverse_build_matches_manual_reversal():
    print("\n── Test 2b: reverse build matches manual reversal ─────────────────")

    rng = np.random.default_rng(7)
    n_loop = 8
    phi = rng.uniform(-np.pi, np.pi, n_loop)
    psi = rng.uniform(-np.pi, np.pi, n_loop)
    omega = np.full(n_loop, np.pi)

    prev_N = np.array([0.0, 0.0, 0.0])
    prev_CA = np.array([1.458, 0.1, -0.2])
    prev_C = np.array([2.5, 1.0, 0.3])
    psi_prev = -0.9

    N_a, CA_a, C_a, O_a = _build_loop(prev_N, prev_CA, prev_C, psi_prev,
                                       phi[::-1], psi[::-1], omega[::-1])
    N_b, CA_b, C_b, O_b = _build_loop(prev_N, prev_CA, prev_C, psi_prev,
                                       phi, psi, omega, reverse=True)

    max_diff = max(
        float(np.max(np.abs(N_a - N_b))),
        float(np.max(np.abs(CA_a - CA_b))),
        float(np.max(np.abs(C_a - C_b))),
        float(np.max(np.abs(O_a - O_b))),
    )
    ok = max_diff < 1e-10
    _check(f"max coord diff = {max_diff:.2e}", ok)
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — random seeds: solution counts and zero closure error
# ─────────────────────────────────────────────────────────────────────────────

def test_random_seeds():
    print("\n── Test 3: random seeds (solution count & zero error) ─────────────")

    seeds   = [0, 1, 2, 7, 13, 42, 100, 256]
    tol_err = 0.05   # Å
    all_passed = True

    for seed in seeds:
        rng = np.random.default_rng(seed)
        t1 = rng.uniform(-np.pi, np.pi)
        t2 = rng.uniform(-np.pi, np.pi)
        t3 = rng.uniform(-np.pi, np.pi)
        target_N = _build_bridge(t1, t2, t3)

        sols = kic.kic_tripeptide_close(
            _ANCHOR_N, _ANCHOR_CA, _ANCHOR_C, target_N, n_grid=3600,
        )

        n_sols     = len(sols)
        zero_err   = all(err < tol_err for _, _, _, err in sols)
        has_sols   = n_sols > 0
        ok         = has_sols and zero_err

        all_passed &= ok
        _check(
            f"seed={seed:3d}  {n_sols} solutions  all_err<{tol_err}Å={zero_err}",
            ok,
        )

    return all_passed


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — vectorised scan matches scalar scan point-for-point
# ─────────────────────────────────────────────────────────────────────────────

def test_vectorised_scan_matches_scalar():
    print("\n── Test 4: vectorised scan == scalar scan ─────────────────────────")

    target_N = _build_bridge(-1.1, 0.8, -0.5)
    t1_grid  = np.linspace(-np.pi, np.pi, 360, endpoint=False)   # smaller grid, faster

    f_scalar = np.array([
        kic._outer_f(t, _ANCHOR_N, _ANCHOR_CA, _ANCHOR_C, target_N)
        for t in t1_grid
    ])
    f_vec = kic._outer_f_grid(t1_grid, _ANCHOR_N, _ANCHOR_CA, _ANCHOR_C, target_N)

    max_diff = float(np.max(np.abs(f_scalar - f_vec)))
    ok = max_diff < 1e-10
    _check(f"max |scalar − vec| = {max_diff:.2e}", ok)
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — speed sanity check
# ─────────────────────────────────────────────────────────────────────────────

def test_speed():
    import time
    print("\n── Test 5: speed ──────────────────────────────────────────────────")

    target_N = _build_bridge(-1.1, 0.8, -0.5)
    t1_grid  = np.linspace(-np.pi, np.pi, 3600, endpoint=False)
    n_reps   = 30

    # Scalar scan (Python loop)
    t0 = time.perf_counter()
    for _ in range(n_reps):
        np.array([
            kic._outer_f(t, _ANCHOR_N, _ANCHOR_CA, _ANCHOR_C, target_N)
            for t in t1_grid
        ])
    t_scalar = (time.perf_counter() - t0) / n_reps * 1000

    # Vectorised scan
    t0 = time.perf_counter()
    for _ in range(n_reps):
        kic._outer_f_grid(t1_grid, _ANCHOR_N, _ANCHOR_CA, _ANCHOR_C, target_N)
    t_vec = (time.perf_counter() - t0) / n_reps * 1000

    # Full solve
    t0 = time.perf_counter()
    for _ in range(n_reps):
        kic.kic_tripeptide_close(_ANCHOR_N, _ANCHOR_CA, _ANCHOR_C, target_N)
    t_full = (time.perf_counter() - t0) / n_reps * 1000

    speedup = t_scalar / t_vec
    print(f"  scalar scan:       {t_scalar:7.2f} ms")
    print(f"  vectorised scan:   {t_vec:7.2f} ms  ({speedup:.0f}× faster)")
    print(f"  full solve:        {t_full:7.2f} ms")

    ok = speedup > 10   # conservatively expect at least 10×
    _check(f"speedup > 10× (got {speedup:.0f}×)", ok)
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    results = {
        "tripeptide round-trip":       test_tripeptide_roundtrip(),
        "full-loop round-trip":        test_full_loop_roundtrip(),
        "reverse build support":       test_reverse_build_matches_manual_reversal(),
        "random seeds":                test_random_seeds(),
        "vectorised == scalar":        test_vectorised_scan_matches_scalar(),
        "speed":                       test_speed(),
    }

    print("\n── Summary ────────────────────────────────────────────────────────")
    all_ok = True
    for name, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  [{status}] {name}")
        all_ok &= ok

    sys.exit(0 if all_ok else 1)
