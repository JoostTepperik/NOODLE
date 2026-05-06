#!/usr/bin/env python3
"""
Validation tests for data processing pipeline
Checks that torsion angles, distances, and data integrity are correct
"""

import numpy as np
from pathlib import Path
from Bio.PDB import PDBParser, calc_dihedral
import matplotlib.pyplot as plt
from collections import defaultdict

from data_processing import TorsionExtractor, TorsionDataset

class ProcessingValidator:
    """Validate torsion angle extraction accuracy"""
    
    def __init__(self):
        self.parser = PDBParser(QUIET=True)
        self.extractor = TorsionExtractor()
    
    def test_known_structure(self, pdb_file):
        """
        Test on a known structure and compare to manual calculations
        """
        print("="*60)
        print("TEST 1: Known Structure Validation")
        print("="*60)
        
        # Extract with our method
        triplets = self.extractor.extract_from_structure(pdb_file)
        print(f"Extracted {len(triplets)} triplets from {pdb_file}")
        
        # Manual verification for first 5 triplets
        structure = self.parser.get_structure('test', pdb_file)
        
        print("\nManual verification of first 5 triplets:")
        verified_count = 0
        
        for model in structure:
            for chain in model:
                residues = [r for r in chain if r.id[0] == ' ']
                
                for i in range(min(5, len(residues) - 2)):
                    res_prev = residues[i]
                    res_curr = residues[i + 1]
                    res_next = residues[i + 2]
                    
                    # Skip if not continuous
                    if not (res_curr.id[1] - res_prev.id[1] == 1 and 
                           res_next.id[1] - res_curr.id[1] == 1):
                        continue
                    
                    # Manual phi calculation
                    try:
                        phi_manual = calc_dihedral(
                            res_prev['C'].get_vector(),
                            res_curr['N'].get_vector(),
                            res_curr['CA'].get_vector(),
                            res_curr['C'].get_vector()
                        )
                        phi_manual_deg = np.degrees(phi_manual)
                    except:
                        continue
                    
                    # Manual psi calculation
                    try:
                        psi_manual = calc_dihedral(
                            res_curr['N'].get_vector(),
                            res_curr['CA'].get_vector(),
                            res_curr['C'].get_vector(),
                            res_next['N'].get_vector()
                        )
                        psi_manual_deg = np.degrees(psi_manual)
                    except:
                        continue
                    
                    # Find corresponding triplet in extracted data
                    matching_triplet = None
                    for t in triplets:
                        if (t['chain_id'] == chain.id and 
                            t['residue_index'] == res_curr.id[1]):
                            matching_triplet = t
                            break
                    
                    if matching_triplet:
                        phi_diff = abs(matching_triplet['phi'] - phi_manual_deg)
                        psi_diff = abs(matching_triplet['psi'] - psi_manual_deg)
                        
                        # Handle periodic boundary (e.g., -179° vs 179°)
                        if phi_diff > 180:
                            phi_diff = 360 - phi_diff
                        if psi_diff > 180:
                            psi_diff = 360 - psi_diff
                        
                        print(f"\nResidue {res_curr.resname} {res_curr.id[1]}:")
                        print(f"  φ: Manual={phi_manual_deg:.2f}°, Extracted={matching_triplet['phi']:.2f}°, Diff={phi_diff:.4f}°")
                        print(f"  ψ: Manual={psi_manual_deg:.2f}°, Extracted={matching_triplet['psi']:.2f}°, Diff={psi_diff:.4f}°")
                        
                        # Check if difference is small (should be < 0.01°)
                        if phi_diff < 0.01 and psi_diff < 0.01:
                            print("  ✓ PASS")
                            verified_count += 1
                        else:
                            print("  ✗ FAIL - angles don't match!")
                            return False
        
        print(f"\n✓ Verified {verified_count} triplets - all angles match!")
        return True
    
    def test_ramachandran_distribution(self, triplets):
        """
        Test that torsion angles follow expected Ramachandran distribution
        """
        print("\n" + "="*60)
        print("TEST 2: Ramachandran Distribution")
        print("="*60)
        
        phis = [t['phi'] for t in triplets]
        psis = [t['psi'] for t in triplets]
        
        print(f"Analyzing {len(triplets)} triplets")
        
        # Check ranges
        assert all(-180 <= phi <= 180 for phi in phis), "Phi angles out of range!"
        assert all(-180 <= psi <= 180 for psi in psis), "Psi angles out of range!"
        print("✓ All angles in valid range [-180, 180]")
        
        # Check for common secondary structure regions
        # α-helix: φ ≈ -60°, ψ ≈ -45°
        alpha_helix = sum(1 for phi, psi in zip(phis, psis) 
                         if -90 < phi < -30 and -70 < psi < -20)
        
        # β-sheet: φ ≈ -120°, ψ ≈ +120°
        beta_sheet = sum(1 for phi, psi in zip(phis, psis)
                        if -150 < phi < -90 and 90 < psi < 150)
        
        # Left-handed helix (rare): φ ≈ +60°, ψ ≈ +60°
        left_handed = sum(1 for phi, psi in zip(phis, psis)
                         if 30 < phi < 90 and 30 < psi < 90)
        
        # Forbidden region (should be very few)
        forbidden = sum(1 for phi, psi in zip(phis, psis)
                       if 0 < phi < 180 and -180 < psi < 0)
        
        total = len(triplets)
        print(f"\nSecondary structure distribution:")
        print(f"  α-helix region: {alpha_helix} ({100*alpha_helix/total:.1f}%)")
        print(f"  β-sheet region: {beta_sheet} ({100*beta_sheet/total:.1f}%)")
        print(f"  Left-handed region: {left_handed} ({100*left_handed/total:.1f}%)")
        print(f"  Forbidden region: {forbidden} ({100*forbidden/total:.1f}%)")
        
        # Sanity checks
        assert alpha_helix > 0.1 * total, "Too few α-helix angles!"
        assert beta_sheet > 0.05 * total, "Too few β-sheet angles!"
        assert forbidden < 0.15 * total, "Too many forbidden region angles!"
        
        print("\n✓ Distribution looks reasonable!")
        return True
    
    def test_ca_distances(self, triplets):
        """
        Test that CA-CA distances are physically reasonable
        """
        print("\n" + "="*60)
        print("TEST 3: CA-CA Distance Validation")
        print("="*60)
        
        distances = []
        for t in triplets:
            distances.append(t['ca_dist_prev'])
            distances.append(t['ca_dist_next'])
        
        distances = [d for d in distances if d is not None]
        
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        print(f"CA-CA distances (Å):")
        print(f"  Mean: {mean_dist:.2f} ± {std_dist:.2f}")
        print(f"  Range: [{min_dist:.2f}, {max_dist:.2f}]")
        
        # Expected: CA-CA distance in peptide bond ≈ 3.8 Å
        # Should be between 3.6-4.0 Å for most cases
        expected_mean = 3.8
        
        assert 3.5 < mean_dist < 4.1, f"Mean distance {mean_dist:.2f} Å is unusual!"
        assert min_dist > 3.0, f"Minimum distance {min_dist:.2f} Å too small!"
        assert max_dist < 5.0, f"Maximum distance {max_dist:.2f} Å too large!"
        
        print(f"\n✓ Expected mean ≈ 3.8 Å, got {mean_dist:.2f} Å - looks good!")
        return True
    
    def test_residue_distribution(self, triplets):
        """
        Test that residue type distribution is reasonable
        """
        print("\n" + "="*60)
        print("TEST 4: Residue Type Distribution")
        print("="*60)
        
        aa_counts = defaultdict(int)
        for t in triplets:
            aa_counts[t['res_curr']] += 1
        
        total = sum(aa_counts.values())
        
        print(f"\nResidue frequencies (top 10):")
        sorted_aa = sorted(aa_counts.items(), key=lambda x: x[1], reverse=True)
        
        for aa, count in sorted_aa[:10]:
            freq = 100 * count / total
            print(f"  {aa}: {count} ({freq:.2f}%)")
        
        # Known abundant amino acids in proteins
        abundant = ['L', 'A', 'G', 'V', 'E', 'S']  # Leu, Ala, Gly, Val, Glu, Ser
        
        # Check that common amino acids are indeed common
        abundant_count = sum(aa_counts[aa] for aa in abundant)
        abundant_freq = 100 * abundant_count / total
        
        print(f"\nAbundant residues (L,A,G,V,E,S): {abundant_freq:.1f}%")
        
        # Should be roughly 40-50% of all residues
        assert abundant_freq > 35, "Abundant residues too rare!"
        assert abundant_freq < 60, "Abundant residues too common!"
        
        print("✓ Distribution looks biologically reasonable!")
        return True
    
    def test_glycine_proline_special_cases(self, triplets):
        """
        Test that Glycine (flexible) and Proline (restricted) show expected behavior
        """
        print("\n" + "="*60)
        print("TEST 5: Glycine and Proline Special Cases")
        print("="*60)
        
        gly_phis = [t['phi'] for t in triplets if t['res_curr'] == 'G']
        gly_psis = [t['psi'] for t in triplets if t['res_curr'] == 'G']
        
        pro_phis = [t['phi'] for t in triplets if t['res_curr'] == 'P']
        pro_psis = [t['psi'] for t in triplets if t['res_curr'] == 'P']
        
        other_phis = [t['phi'] for t in triplets if t['res_curr'] not in ['G', 'P']]
        other_psis = [t['psi'] for t in triplets if t['res_curr'] not in ['G', 'P']]
        
        print(f"\nGlycine (flexible):")
        print(f"  Count: {len(gly_phis)}")
        print(f"  φ std: {np.std(gly_phis):.1f}°")
        print(f"  ψ std: {np.std(gly_psis):.1f}°")
        
        print(f"\nProline (restricted):")
        print(f"  Count: {len(pro_phis)}")
        print(f"  φ mean: {np.mean(pro_phis):.1f}° (expected ≈ -60°)")
        print(f"  φ std: {np.std(pro_phis):.1f}°")
        
        print(f"\nOther residues:")
        print(f"  φ std: {np.std(other_phis):.1f}°")
        print(f"  ψ std: {np.std(other_psis):.1f}°")
        
        # Glycine should be more flexible (higher std dev)
        assert np.std(gly_phis) > np.std(other_phis), "Glycine φ not more flexible!"
        print("✓ Glycine shows expected flexibility")
        
        # Proline φ should be around -60°
        pro_phi_mean = np.mean(pro_phis)
        assert -75 < pro_phi_mean < -45, f"Proline φ mean unusual: {pro_phi_mean:.1f}°"
        print("✓ Proline shows expected restriction")
        
        return True
    
    def test_omega_angles(self, triplets):
        """
        Test that omega angles (peptide bond planarity) are near 180° or 0°
        """
        print("\n" + "="*60)
        print("TEST 6: Omega Angle (Peptide Bond Planarity)")
        print("="*60)
        
        omegas = [t['omega'] for t in triplets]
        
        # Count trans (≈180°) vs cis (≈0°) peptide bonds
        trans = sum(1 for o in omegas if abs(o - 180) < 30 or abs(o + 180) < 30)
        cis = sum(1 for o in omegas if abs(o) < 30)
        
        trans_pct = 100 * trans / len(omegas)
        cis_pct = 100 * cis / len(omegas)
        
        print(f"Peptide bond conformations:")
        print(f"  Trans (ω ≈ 180°): {trans} ({trans_pct:.1f}%)")
        print(f"  Cis (ω ≈ 0°): {cis} ({cis_pct:.1f}%)")
        
        # Most peptide bonds should be trans (>95%)
        assert trans_pct > 90, "Too few trans peptide bonds!"
        assert cis_pct < 10, "Too many cis peptide bonds!"
        
        print("✓ Peptide bonds mostly planar (trans)")
        return True
    
    def plot_ramachandran(self, triplets, output_file='ramachandran_validation.png'):
        """
        Generate Ramachandran plot for visual inspection
        """
        print("\n" + "="*60)
        print("Generating Ramachandran Plot")
        print("="*60)
        
        phis = [t['phi'] for t in triplets]
        psis = [t['psi'] for t in triplets]
        
        plt.figure(figsize=(8, 8))
        plt.hexbin(phis, psis, gridsize=50, cmap='Blues', mincnt=1)
        plt.colorbar(label='Count')
        
        plt.xlabel('φ (degrees)', fontsize=12)
        plt.ylabel('ψ (degrees)', fontsize=12)
        plt.title(f'Ramachandran Plot ({len(triplets)} residues)', fontsize=14)
        
        plt.xlim(-180, 180)
        plt.ylim(-180, 180)
        plt.axhline(0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
        plt.axvline(0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        print(f"✓ Saved Ramachandran plot to {output_file}")
        plt.close()
    
    def run_all_tests(self, pdb_file):
        """
        Run complete validation suite
        """
        print("\n" + "#"*60)
        print("# RUNNING COMPLETE VALIDATION SUITE")
        print("#"*60 + "\n")
        
        # Extract triplets
        triplets = self.extractor.extract_from_structure(pdb_file)
        
        if len(triplets) == 0:
            print("ERROR: No triplets extracted!")
            return False
        
        print(f"Testing with {len(triplets)} triplets from {pdb_file}\n")
        
        # Run all tests
        tests = [
            self.test_known_structure(pdb_file),
            self.test_ramachandran_distribution(triplets),
            self.test_ca_distances(triplets),
            self.test_residue_distribution(triplets),
            self.test_glycine_proline_special_cases(triplets),
            self.test_omega_angles(triplets),
        ]
        
        # Generate plot
        self.plot_ramachandran(triplets)
        
        # Summary
        print("\n" + "#"*60)
        print("# VALIDATION SUMMARY")
        print("#"*60)
        
        if all(tests):
            print("✓ ALL TESTS PASSED!")
            print("\nYour data processing pipeline is working correctly.")
            return True
        else:
            print("✗ SOME TESTS FAILED!")
            print("\nPlease review the errors above.")
            return False


def main():
    """
    Run validation on test structure
    """
    import sys
    
    # Check if test structure exists
    test_pdb = Path('data/test/pdb_files/1ubq_redo.pdb')
    
    if not test_pdb.exists():
        # Try alternative location
        test_pdb = Path('data/test/pdb_files/1ubq.pdb')
    
    if not test_pdb.exists():
        print(f"ERROR: Test structure not found at {test_pdb}")
        print("\nPlease run: python scripts/test_pipeline.py first")
        sys.exit(1)
    
    # Run validation
    validator = ProcessingValidator()
    success = validator.run_all_tests(test_pdb)
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()