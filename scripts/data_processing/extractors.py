import numpy as np
from pathlib import Path
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1

class SevenMerTorsionExtractor:
    """Extract 7-mer (±3 context) torsion angle features"""
    
    def __init__(self):
        self.parser = PDBParser(QUIET=True)
        
        # Amino acid to index mapping
        self.aa_to_idx = {
            'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
            'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
            'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
            'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
        }
    
    def extract_from_structure(self, pdb_file):
        """
        Extract 7-mer features from a PDB file
        
        Returns:
            List of dicts, each containing 7-mer features
        """
        pdb_id = Path(pdb_file).stem.split('_')[0]
        structure = self.parser.get_structure(pdb_id, pdb_file)
        
        all_features = []
        
        # Process each chain
        for chain in structure.get_chains():
            chain_features = self._extract_from_chain(chain, pdb_id)
            all_features.extend(chain_features)
        
        return all_features
    
    def _extract_from_chain(self, chain, pdb_id):
        """Extract 7-mer features from a single chain"""
        
        # Get all standard amino acid residues with CA atoms
        residues = []
        for res in chain:
            if is_aa(res) and 'CA' in res:
                residues.append(res)
        
        if len(residues) < 7:  # Need at least 7 residues
            return []
        
        features = []
        
        # Extract 7-mers (need ±3 neighbors)
        for i in range(3, len(residues) - 3):
            try:
                feature = self._extract_septamer(residues, i, pdb_id, chain.id)
                if feature is not None:
                    features.append(feature)
            except Exception as e:
                continue  # Skip problematic residues
        
        return features
    
    def _extract_septamer(self, residues, i, pdb_id, chain_id):
        """
        Extract features for a 7-mer centered at position i
        
        Context: res[i-3], res[i-2], res[i-1], res[i], res[i+1], res[i+2], res[i+3]
        """
        # Get 7 consecutive residues
        res_i_minus_3 = residues[i - 3]
        res_i_minus_2 = residues[i - 2]
        res_i_minus_1 = residues[i - 1]
        res_i = residues[i]
        res_i_plus_1 = residues[i + 1]
        res_i_plus_2 = residues[i + 2]
        res_i_plus_3 = residues[i + 3]
        
        # Get residue types
        try:
            res_type_i_minus_3 = self.aa_to_idx[seq1(res_i_minus_3.get_resname())]
            res_type_i_minus_2 = self.aa_to_idx[seq1(res_i_minus_2.get_resname())]
            res_type_i_minus_1 = self.aa_to_idx[seq1(res_i_minus_1.get_resname())]
            res_type_i = self.aa_to_idx[seq1(res_i.get_resname())]
            res_type_i_plus_1 = self.aa_to_idx[seq1(res_i_plus_1.get_resname())]
            res_type_i_plus_2 = self.aa_to_idx[seq1(res_i_plus_2.get_resname())]
            res_type_i_plus_3 = self.aa_to_idx[seq1(res_i_plus_3.get_resname())]
        except (KeyError, ValueError):
            return None  # Non-standard amino acid
        
        # Calculate torsion angles for center residue (i)
        phi = self._calculate_phi(res_i_minus_1, res_i)
        psi = self._calculate_psi(res_i, res_i_plus_1)
        omega = self._calculate_omega(res_i, res_i_plus_1)
        
        if phi is None or psi is None:
            return None
        
        # Calculate CA-CA distances (6 consecutive distances)
        ca_dists = []
        for j in range(-3, 3):
            ca_prev = residues[i + j]['CA'].get_coord()
            ca_next = residues[i + j + 1]['CA'].get_coord()
            dist = np.linalg.norm(ca_next - ca_prev)
            ca_dists.append(dist)
        
        return {
            'pdb_id': pdb_id,
            'chain_id': chain_id,
            'residue_index': i,
            
            # 7 residue types
            'res_i_minus_3': res_type_i_minus_3,
            'res_i_minus_2': res_type_i_minus_2,
            'res_i_minus_1': res_type_i_minus_1,
            'res_i': res_type_i,
            'res_i_plus_1': res_type_i_plus_1,
            'res_i_plus_2': res_type_i_plus_2,
            'res_i_plus_3': res_type_i_plus_3,
            
            # Torsion angles for center residue
            'phi': phi,
            'psi': psi,
            'omega': omega,
            
            # 6 CA-CA distances
            'ca_dist_i_minus_3': ca_dists[0],
            'ca_dist_i_minus_2': ca_dists[1],
            'ca_dist_i_minus_1': ca_dists[2],
            'ca_dist_i': ca_dists[3],
            'ca_dist_i_plus_1': ca_dists[4],
            'ca_dist_i_plus_2': ca_dists[5],
        }
    
    def _calculate_phi(self, res_prev, res_curr):
        """Calculate phi angle: C(i-1) - N(i) - CA(i) - C(i)"""
        try:
            c_prev = res_prev['C'].get_vector()
            n_curr = res_curr['N'].get_vector()
            ca_curr = res_curr['CA'].get_vector()
            c_curr = res_curr['C'].get_vector()
            
            phi = self._calc_dihedral(c_prev, n_curr, ca_curr, c_curr)
            return phi
        except KeyError:
            return None
    
    def _calculate_psi(self, res_curr, res_next):
        """Calculate psi angle: N(i) - CA(i) - C(i) - N(i+1)"""
        try:
            n_curr = res_curr['N'].get_vector()
            ca_curr = res_curr['CA'].get_vector()
            c_curr = res_curr['C'].get_vector()
            n_next = res_next['N'].get_vector()
            
            psi = self._calc_dihedral(n_curr, ca_curr, c_curr, n_next)
            return psi
        except KeyError:
            return None
    
    def _calculate_omega(self, res_curr, res_next):
        """Calculate omega angle: CA(i) - C(i) - N(i+1) - CA(i+1)"""
        try:
            ca_curr = res_curr['CA'].get_vector()
            c_curr = res_curr['C'].get_vector()
            n_next = res_next['N'].get_vector()
            ca_next = res_next['CA'].get_vector()
            
            omega = self._calc_dihedral(ca_curr, c_curr, n_next, ca_next)
            return omega
        except KeyError:
            return None
    
    def _calc_dihedral(self, p1, p2, p3, p4):
        """
        Calculate dihedral angle between 4 points in degrees
        
        Args:
            p1, p2, p3, p4: Bio.PDB.Vector objects
        
        Returns:
            Dihedral angle in degrees [-180, 180]
        """
        # Vector from p1 to p2, p2 to p3, p3 to p4
        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3
        
        # Normal vectors to planes
        n1 = b1 ** b2  # Cross product
        n2 = b2 ** b3
        
        # Normalize
        n1_norm = n1.norm()
        n2_norm = n2.norm()
        
        if n1_norm < 1e-6 or n2_norm < 1e-6:
            return None  # Degenerate case
        
        n1 = n1 / n1_norm
        n2 = n2 / n2_norm
        
        # Calculate angle
        m1 = n1 ** (b2 / b2.norm())
        
        x = n1 * n2
        y = m1 * n2
        
        angle = np.degrees(np.arctan2(y, x))
        
        return angle


