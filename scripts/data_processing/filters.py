from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.DSSP import DSSP
import numpy as np

class StructureFilter:
    """Filter structures by quality metrics"""
    
    def __init__(self, resolution_cutoff=2.5, min_length=50, max_length=1000):
        self.resolution_cutoff = resolution_cutoff
        self.min_length = min_length
        self.max_length = max_length
        self.parser = PDBParser(QUIET=True)
    
    def parse_pdb(self, pdb_file):
        """Load structure and extract metadata"""
        structure = self.parser.get_structure('protein', pdb_file)
        
        # Get header information
        with open(pdb_file) as f:
            lines = f.readlines()
        
        metadata = {
            'resolution': self._extract_resolution(lines),
            'method': self._extract_method(lines),
            'r_free': self._extract_rfree(lines),
            'structure': structure
        }
        
        return metadata
    
    def _extract_resolution(self, lines):
        """Extract resolution from PDB header"""
        for line in lines:
            if line.startswith('REMARK   2 RESOLUTION.'):
                try:
                    res = float(line.split()[3])
                    return res
                except:
                    return None
        return None
    
    def _extract_method(self, lines):
        """Extract experimental method"""
        for line in lines:
            if line.startswith('EXPDTA'):
                return line[10:].strip()
        return None
    
    def _extract_rfree(self, lines):
        """Extract R-free value"""
        for line in lines:
            if 'FREE R VALUE' in line:
                try:
                    rfree = float(line.split(':')[-1].strip())
                    return rfree
                except:
                    return None
        return None
    
    def check_quality(self, metadata):
        """Apply quality filters"""
        filters_passed = {}
        
        # Resolution filter
        if metadata['resolution'] is None:
            filters_passed['resolution'] = False
        else:
            filters_passed['resolution'] = metadata['resolution'] <= self.resolution_cutoff
        
        # Method filter (X-ray only)
        if metadata['method'] is None:
            filters_passed['method'] = False
        else:
            filters_passed['method'] = 'X-RAY' in metadata['method'].upper()
        
        # R-free filter (optional but recommended)
        if metadata['r_free'] is not None:
            filters_passed['r_free'] = metadata['r_free'] <= 0.25
        else:
            filters_passed['r_free'] = True  # Missing R-free is ok
        
        # Chain length filter
        structure = metadata['structure']
        chain_lengths = []
        for model in structure:
            for chain in model:
                residues = [r for r in chain if r.id[0] == ' ']  # Exclude HETATM
                chain_lengths.append(len(residues))
        
        if chain_lengths:
            max_chain_len = max(chain_lengths)
            filters_passed['length'] = (
                self.min_length <= max_chain_len <= self.max_length
            )
        else:
            filters_passed['length'] = False
        
        return filters_passed, all(filters_passed.values())
    
    def check_completeness(self, structure):
        """Check for missing residues, atoms"""
        issues = []
        
        for model in structure:
            for chain in model:
                residues = [r for r in chain if r.id[0] == ' ']
                
                # Check for missing backbone atoms
                for residue in residues:
                    required_atoms = ['N', 'CA', 'C', 'O']
                    missing = [a for a in required_atoms if a not in residue]
                    
                    if missing:
                        issues.append({
                            'type': 'missing_atoms',
                            'residue': residue.id,
                            'atoms': missing
                        })
                
                # Check for gaps in sequence (missing residues)
                res_ids = [r.id[1] for r in residues]
                if len(res_ids) > 1:
                    gaps = []
                    for i in range(len(res_ids) - 1):
                        if res_ids[i+1] - res_ids[i] > 1:
                            gaps.append((res_ids[i], res_ids[i+1]))
                    
                    if gaps:
                        issues.append({
                            'type': 'sequence_gaps',
                            'gaps': gaps
                        })
        
        return issues

# Usage
'''filter_obj = StructureFilter(resolution_cutoff=2.5)
metadata = filter_obj.parse_pdb('data/pdb_redo/1abc_redo.pdb')
filters, passed = filter_obj.check_quality(metadata)

if passed:
    issues = filter_obj.check_completeness(metadata['structure'])
    if not issues:
        print("Structure passes all quality checks!")'''