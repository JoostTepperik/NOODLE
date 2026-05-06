"""
Comprehensive diagnostic for frame-based loop modeling.

Usage:
    python diagnose_loop.py <predicted.pdb> <native.pdb> <loop_start> <loop_end>

    loop_start / loop_end: 1-based residue indices (inclusive), e.g. 3 17

Checks:
  1. Flank integrity   – are native flank coords unchanged in the prediction?
  2. Bond lengths      – N-CA, CA-C, C-N(peptide) for every residue
  3. Bond angles       – N-CA-C, CA-C-N, C-N-CA for every residue
  4. Omega angles      – peptide planarity (should be ~180°)
  5. Phi / psi angles  – Ramachandran plot + distance from nearest basin
  6. Closure bonds     – the two junction bonds flank→loop and loop→flank
  7. CA-RMSD           – per-residue and loop-only RMSD vs native
  8. Summary table     – one line per residue
"""

import sys
import numpy as np
import os

# ─────────────────────────────────────────────────────────────────────────────
# Ideal geometry reference values
# ─────────────────────────────────────────────────────────────────────────────

IDEAL = {
    'N_CA':   1.458,   # Å
    'CA_C':   1.525,   # Å
    'C_N':    1.329,   # Å  peptide bond
    'N_CA_C': 111.2,   # °  tau angle
    'CA_C_N': 116.2,   # °
    'C_N_CA': 121.7,   # °
    'omega':  180.0,   # °  trans peptide
}

# Ramachandran basin centres (phi, psi) in degrees
RAMA_BASINS = {
    'alpha_R': (-57,  -47),
    'beta':    (-119, +113),
    'alpha_L': (+57,  +47),
    'pp_II':   (-78,  +149),
}

WARN_BOND   = 0.15   # Å  deviation from ideal before warning
WARN_ANGLE  = 5.0    # °  deviation from ideal before warning
WARN_OMEGA  = 15.0   # °  deviation from 180° before warning
WARN_FLANK  = 0.01   # Å  max allowed CA drift in flank residues


# ─────────────────────────────────────────────────────────────────────────────
# PDB loading
# ─────────────────────────────────────────────────────────────────────────────

THREE_TO_ONE = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}


def load_pdb(path):
    """Return seq, N, CA, C, O as arrays. Handles multi-model (takes MODEL 1)."""
    N_l, CA_l, C_l, O_l, seq = [], [], [], [], []
    in_model1 = True

    with open(path) as f:
        for line in f:
            if line.startswith('MODEL'):
                model_num = int(line.split()[1])
                in_model1 = (model_num == 1)
            if line.startswith('ENDMDL') and not in_model1:
                break
            if not in_model1:
                continue
            if not line.startswith('ATOM'):
                continue
            atom = line[12:16].strip()
            res3 = line[17:20].strip()
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])

            if atom == 'CA':
                seq.append(THREE_TO_ONE.get(res3, 'X'))
            if   atom == 'N':  N_l .append([x, y, z])
            elif atom == 'CA': CA_l.append([x, y, z])
            elif atom == 'C':  C_l .append([x, y, z])
            elif atom == 'O':  O_l .append([x, y, z])

    n = len(CA_l)
    O_arr = np.array(O_l) if len(O_l) == n else np.zeros((n, 3))

    return (
        ''.join(seq),
        np.array(N_l),
        np.array(CA_l),
        np.array(C_l),
        O_arr,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def bond(a, b):
    return np.linalg.norm(a - b)


def angle_deg(a, b, c):
    """Angle at vertex b, in degrees."""
    u = a - b;  u /= (np.linalg.norm(u) + 1e-8)
    v = c - b;  v /= (np.linalg.norm(v) + 1e-8)
    return np.degrees(np.arccos(np.clip(np.dot(u, v), -1, 1)))


def dihedral_deg(p1, p2, p3, p4):
    b1 = p2 - p1;  b2 = p3 - p2;  b3 = p4 - p3
    n1 = np.cross(b1, b2);  n2 = np.cross(b2, b3)
    n1 /= (np.linalg.norm(n1) + 1e-8)
    n2 /= (np.linalg.norm(n2) + 1e-8)
    m1  = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-8))
    return np.degrees(np.arctan2(np.dot(m1, n2), np.dot(n1, n2)))


def nearest_basin(phi, psi):
    best, best_d = None, 1e9
    for name, (cp, cs) in RAMA_BASINS.items():
        dphi = abs(((phi - cp) + 180) % 360 - 180)
        dpsi = abs(((psi - cs) + 180) % 360 - 180)
        d    = np.sqrt(dphi**2 + dpsi**2)
        if d < best_d:
            best, best_d = name, d
    return best, best_d


# ─────────────────────────────────────────────────────────────────────────────
# Checks
# ─────────────────────────────────────────────────────────────────────────────

def check_flank_integrity(pred_CA, native_CA, loop_start, loop_end, seq):
    """Check that flank residues are unchanged."""
    print("\n" + "="*70)
    print("1. FLANK INTEGRITY  (CA drift from native)")
    print("="*70)
    n = len(seq)
    flank_idx = list(range(loop_start)) + list(range(loop_end, n))
    max_drift = 0.0
    any_warn  = False
    for i in flank_idx:
        d = np.linalg.norm(pred_CA[i] - native_CA[i])
        max_drift = max(max_drift, d)
        flag = " ← WARN" if d > WARN_FLANK else ""
        if d > WARN_FLANK:
            any_warn = True
        print(f"  Res {i+1:3d} {seq[i]} (flank): CA drift = {d:.6f} Å{flag}")
    if not any_warn:
        print(f"  ✓ All flank residues intact (max drift {max_drift:.2e} Å)")
    else:
        print(f"  ✗ Flank residues have drifted — flanks are NOT fixed at native!")


def check_bond_lengths(N, CA, C, seq, loop_start, loop_end, label):
    """Check N-CA, CA-C, and C-N(peptide) bond lengths."""
    print(f"\n{'='*70}")
    print(f"2. BOND LENGTHS  ({label})")
    print(f"{'='*70}")
    print(f"  {'Res':>4}  {'AA':>2}  {'Region':>6}  "
          f"{'N-CA':>7}  {'CA-C':>7}  {'C-N(+1)':>8}  Notes")
    print(f"  {'-'*62}")

    n       = len(seq)
    bad_any = False

    for i in range(n):
        region = 'LOOP' if loop_start <= i < loop_end else 'flank'
        d_nca  = bond(N[i],  CA[i])
        d_cac  = bond(CA[i], C[i])
        d_cn   = bond(C[i],  N[i+1]) if i < n-1 else float('nan')

        notes = []
        if abs(d_nca - IDEAL['N_CA'])  > WARN_BOND: notes.append(f"N-CA {d_nca-IDEAL['N_CA']:+.3f}")
        if abs(d_cac - IDEAL['CA_C'])  > WARN_BOND: notes.append(f"CA-C {d_cac-IDEAL['CA_C']:+.3f}")
        if not np.isnan(d_cn):
            if abs(d_cn  - IDEAL['C_N'])   > WARN_BOND: notes.append(f"C-N {d_cn-IDEAL['C_N']:+.3f}")

        flag = " ← WARN" if notes else ""
        if notes: bad_any = True

        cn_str = f"{d_cn:7.3f}" if not np.isnan(d_cn) else "    n/a"
        print(f"  {i+1:4d}  {seq[i]:>2}  {region:>6}  "
              f"{d_nca:7.3f}  {d_cac:7.3f}  {cn_str}  {','.join(notes)}{flag}")

    print(f"\n  Ideal: N-CA={IDEAL['N_CA']:.3f}  CA-C={IDEAL['CA_C']:.3f}  C-N={IDEAL['C_N']:.3f}  (warn > ±{WARN_BOND} Å)")
    if not bad_any:
        print("  ✓ All bond lengths within tolerance")


def check_bond_angles(N, CA, C, seq, loop_start, loop_end, label):
    """Check N-CA-C (tau), CA-C-N, C-N-CA bond angles."""
    print(f"\n{'='*70}")
    print(f"3. BOND ANGLES  ({label})")
    print(f"{'='*70}")
    print(f"  {'Res':>4}  {'AA':>2}  {'Region':>6}  "
          f"{'N-CA-C':>8}  {'CA-C-N':>8}  {'C-N-CA':>8}  Notes")
    print(f"  {'-'*66}")

    n = len(seq)
    bad_any = False

    for i in range(n):
        region  = 'LOOP' if loop_start <= i < loop_end else 'flank'
        a_tau   = angle_deg(N[i],  CA[i], C[i])
        a_cacn  = angle_deg(CA[i], C[i],  N[i+1]) if i < n-1 else float('nan')
        a_cnca  = angle_deg(C[i-1], N[i], CA[i])  if i > 0   else float('nan')

        notes = []
        if abs(a_tau  - IDEAL['N_CA_C']) > WARN_ANGLE: notes.append(f"tau {a_tau-IDEAL['N_CA_C']:+.1f}°")
        if not np.isnan(a_cacn):
            if abs(a_cacn - IDEAL['CA_C_N']) > WARN_ANGLE: notes.append(f"CA-C-N {a_cacn-IDEAL['CA_C_N']:+.1f}°")
        if not np.isnan(a_cnca):
            if abs(a_cnca - IDEAL['C_N_CA']) > WARN_ANGLE: notes.append(f"C-N-CA {a_cnca-IDEAL['C_N_CA']:+.1f}°")

        flag = " ← WARN" if notes else ""
        if notes: bad_any = True

        tau_s  = f"{a_tau:8.2f}"
        cacn_s = f"{a_cacn:8.2f}" if not np.isnan(a_cacn) else "     n/a"
        cnca_s = f"{a_cnca:8.2f}" if not np.isnan(a_cnca) else "     n/a"
        print(f"  {i+1:4d}  {seq[i]:>2}  {region:>6}  "
              f"{tau_s}  {cacn_s}  {cnca_s}  {','.join(notes)}{flag}")

    print(f"\n  Ideal: N-CA-C={IDEAL['N_CA_C']:.1f}°  CA-C-N={IDEAL['CA_C_N']:.1f}°  "
          f"C-N-CA={IDEAL['C_N_CA']:.1f}°  (warn > ±{WARN_ANGLE}°)")
    if not bad_any:
        print("  ✓ All bond angles within tolerance")


def check_omega(N, CA, C, seq, loop_start, loop_end, label):
    """Check omega (peptide planarity). Should be ~180° (trans) or ~0° (cis)."""
    print(f"\n{'='*70}")
    print(f"4. OMEGA ANGLES (peptide planarity)  ({label})")
    print(f"{'='*70}")
    print(f"  {'Bond':>10}  {'Region':>6}  {'Omega':>8}  {'|dev|':>7}  Notes")
    print(f"  {'-'*52}")

    n       = len(seq)
    bad_any = False

    for i in range(n - 1):
        region = 'LOOP' if loop_start <= i < loop_end else 'flank'
        omega  = dihedral_deg(CA[i], C[i], N[i+1], CA[i+1])
        dev    = abs(abs(omega) - 180.0)
        flag   = " ← WARN" if dev > WARN_OMEGA else ""
        if dev > WARN_OMEGA: bad_any = True
        print(f"  {seq[i]}{i+1}-{seq[i+1]}{i+2:>4}  {region:>6}  "
              f"{omega:8.2f}  {dev:7.2f}°{flag}")

    print(f"\n  Ideal: ±180° (trans peptide)  (warn if |dev| > {WARN_OMEGA}°)")
    if not bad_any:
        print("  ✓ All peptide bonds trans")


def check_torsions(N, CA, C, seq, loop_start, loop_end, label):
    """Check phi/psi: values, Ramachandran basin, and distance from nearest basin."""
    print(f"\n{'='*70}")
    print(f"5. PHI/PSI TORSION ANGLES  ({label})")
    print(f"{'='*70}")
    print(f"  {'Res':>4}  {'AA':>2}  {'Region':>6}  "
          f"{'phi':>8}  {'psi':>8}  {'Basin':>8}  {'d_basin':>8}  Notes")
    print(f"  {'-'*72}")

    n       = len(seq)
    bad_any = False

    for i in range(n):
        region = 'LOOP' if loop_start <= i < loop_end else 'flank'

        phi = dihedral_deg(C[i-1], N[i], CA[i], C[i]) if i > 0   else float('nan')
        psi = dihedral_deg(N[i], CA[i], C[i], N[i+1]) if i < n-1 else float('nan')

        if np.isnan(phi) or np.isnan(psi):
            phi_s   = "     n/a"
            psi_s   = "     n/a"
            basin_s = "     n/a"
            db_s    = "     n/a"
        else:
            basin, d_basin = nearest_basin(phi, psi)
            phi_s   = f"{phi:8.1f}"
            psi_s   = f"{psi:8.1f}"
            basin_s = f"{basin:>8}"
            db_s    = f"{d_basin:8.1f}"
            flag    = " ← FAR" if d_basin > 60 else ""
            if d_basin > 60 and region == 'LOOP':
                bad_any = True

        print(f"  {i+1:4d}  {seq[i]:>2}  {region:>6}  "
              f"{phi_s}  {psi_s}  {basin_s}  {db_s}")

    print(f"\n  Basin reference: alpha_R(-57,-47)  beta(-119,+113)  "
          f"alpha_L(+57,+47)  pp_II(-78,+149)")
    print(f"  d_basin = sqrt(dphi² + dpsi²)  (warn > 60° for loop residues)")
    if not bad_any:
        print("  ✓ All loop residues near a Ramachandran basin")


def check_closure_bonds(pred_N, pred_CA, pred_C,
                         native_N, native_CA, native_C,
                         loop_start, loop_end, seq):
    """Check the two junction bonds and their native counterparts."""
    print(f"\n{'='*70}")
    print(f"6. CLOSURE / JUNCTION BONDS")
    print(f"{'='*70}")

    ls, le = loop_start, loop_end
    n      = len(seq)

    # N-terminal junction: C of last flank_before → N of loop[0]
    if ls > 0:
        pred_d1   = bond(pred_C[ls-1],   pred_N[ls])
        native_d1 = bond(native_C[ls-1], native_N[ls])
        dev1      = pred_d1 - IDEAL['C_N']
        flag1     = " ← BROKEN" if abs(dev1) > WARN_BOND else ""
        print(f"  N-terminal junction  C({seq[ls-1]}{ls}) → N({seq[ls]}{ls+1})")
        print(f"    Predicted : {pred_d1:.4f} Å  (ideal {IDEAL['C_N']:.3f}, dev {dev1:+.4f}){flag1}")
        print(f"    Native    : {native_d1:.4f} Å")

    # C-terminal junction: C of loop[-1] → N of first flank_after
    if le < n:
        pred_d2   = bond(pred_C[le-1],   pred_N[le])
        native_d2 = bond(native_C[le-1], native_N[le])
        dev2      = pred_d2 - IDEAL['C_N']
        flag2     = " ← BROKEN" if abs(dev2) > WARN_BOND else ""
        print(f"\n  C-terminal junction  C({seq[le-1]}{le}) → N({seq[le]}{le+1})")
        print(f"    Predicted : {pred_d2:.4f} Å  (ideal {IDEAL['C_N']:.3f}, dev {dev2:+.4f}){flag2}")
        print(f"    Native    : {native_d2:.4f} Å")


def check_rmsd(pred_CA, native_CA, seq, loop_start, loop_end):
    """Per-residue CA RMSD and loop-only RMSD."""
    print(f"\n{'='*70}")
    print(f"7. CA-RMSD vs NATIVE")
    print(f"{'='*70}")

    n = len(seq)
    per_res = np.array([bond(pred_CA[i], native_CA[i]) for i in range(n)])

    print(f"  {'Res':>4}  {'AA':>2}  {'Region':>6}  {'CA dist':>9}")
    print(f"  {'-'*38}")
    for i in range(n):
        region = 'LOOP' if loop_start <= i < loop_end else 'flank'
        flag   = " ← LARGE" if (region == 'LOOP' and per_res[i] > 2.0) else ""
        print(f"  {i+1:4d}  {seq[i]:>2}  {region:>6}  {per_res[i]:9.3f} Å{flag}")

    loop_rmsd  = np.sqrt(np.mean(per_res[loop_start:loop_end]**2))
    flank_rmsd = np.sqrt(np.mean(np.concatenate([per_res[:loop_start],
                                                   per_res[loop_end:]])**2))
    total_rmsd = np.sqrt(np.mean(per_res**2))

    print(f"\n  Loop  CA-RMSD  : {loop_rmsd:.3f} Å  ({loop_end-loop_start} residues)")
    print(f"  Flank CA-RMSD  : {flank_rmsd:.4f} Å  (should be ~0 if flanks fixed)")
    print(f"  Total CA-RMSD  : {total_rmsd:.3f} Å")


def summary(pred_N, pred_CA, pred_C,
             native_N, native_CA, native_C,
             seq, loop_start, loop_end):
    """One-line-per-residue summary table."""
    print(f"\n{'='*70}")
    print(f"8. SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"  {'Res':>4}  {'AA':>2}  {'Region':>6}  "
          f"{'CA-dist':>8}  {'N-CA':>6}  {'CA-C':>6}  {'C-N':>6}  "
          f"{'phi':>7}  {'psi':>7}  {'basin':>8}  {'d_bas':>6}")
    print(f"  {'-'*86}")

    n = len(seq)
    for i in range(n):
        region  = 'LOOP' if loop_start <= i < loop_end else 'flank'
        ca_dist = bond(pred_CA[i], native_CA[i])
        d_nca   = bond(pred_N[i],  pred_CA[i])
        d_cac   = bond(pred_CA[i], pred_C[i])
        d_cn    = bond(pred_C[i],  pred_N[i+1]) if i < n-1 else float('nan')
        phi     = dihedral_deg(pred_C[i-1], pred_N[i], pred_CA[i], pred_C[i]) if i > 0 else float('nan')
        psi     = dihedral_deg(pred_N[i], pred_CA[i], pred_C[i], pred_N[i+1]) if i < n-1 else float('nan')

        if not (np.isnan(phi) or np.isnan(psi)):
            basin, d_bas = nearest_basin(phi, psi)
            phi_s  = f"{phi:7.1f}"
            psi_s  = f"{psi:7.1f}"
            bas_s  = f"{basin:>8}"
            dbs_s  = f"{d_bas:6.1f}"
        else:
            phi_s = psi_s = "    n/a"
            bas_s = "     n/a"
            dbs_s = "   n/a"

        cn_s = f"{d_cn:6.3f}" if not np.isnan(d_cn) else "   n/a"

        print(f"  {i+1:4d}  {seq[i]:>2}  {region:>6}  "
              f"{ca_dist:8.3f}  {d_nca:6.3f}  {d_cac:6.3f}  {cn_s}  "
              f"{phi_s}  {psi_s}  {bas_s}  {dbs_s}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def diagnose(pred_pdb, native_pdb, loop_start_1, loop_end_1):
    """
    Run all diagnostics.

    loop_start_1, loop_end_1: 1-based inclusive residue indices.
    """
    # Convert to 0-based half-open [loop_start, loop_end)
    ls = loop_start_1 - 1
    le = loop_end_1       # exclusive

    print(f"\n{'#'*70}")
    print(f"# LOOP MODELING DIAGNOSTIC")
    print(f"#   Predicted : {pred_pdb}")
    print(f"#   Native    : {native_pdb}")
    print(f"#   Loop      : residues {loop_start_1}–{loop_end_1} (1-based, {le-ls} residues)")
    print(f"{'#'*70}")

    seq_p, N_p, CA_p, C_p, O_p = load_pdb(pred_pdb)
    seq_n, N_n, CA_n, C_n, O_n = load_pdb(native_pdb)

    print(f"\n  Predicted sequence : {seq_p}")
    print(f"  Native    sequence : {seq_n}")

    if seq_p != seq_n:
        print(f"\n  ⚠ SEQUENCE MISMATCH — structures may not be comparable!")

    if len(seq_p) != len(seq_n):
        print(f"\n  ✗ LENGTH MISMATCH: predicted={len(seq_p)}, native={len(seq_n)}")
        print(f"    Cannot run diagnostics — check PDB files.")
        return

    check_flank_integrity(CA_p, CA_n, ls, le, seq_p)
    check_bond_lengths(N_p, CA_p, C_p, seq_p, ls, le, "PREDICTED")
    check_bond_angles (N_p, CA_p, C_p, seq_p, ls, le, "PREDICTED")
    check_omega       (N_p, CA_p, C_p, seq_p, ls, le, "PREDICTED")
    check_torsions    (N_p, CA_p, C_p, seq_p, ls, le, "PREDICTED")
    check_closure_bonds(N_p, CA_p, C_p, N_n, CA_n, C_n, ls, le, seq_p)
    check_rmsd        (CA_p, CA_n, seq_p, ls, le)
    summary           (N_p, CA_p, C_p, N_n, CA_n, C_n, seq_p, ls, le)

    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC COMPLETE")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(1)

    pred_pdb   = sys.argv[1]
    native_pdb = sys.argv[2]
    ls1        = int(sys.argv[3])
    le1        = int(sys.argv[4])

    diagnose(pred_pdb, native_pdb, ls1, le1)