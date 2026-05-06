#!/usr/bin/env python3
"""
Query RCSB PDB for high-quality structures
More reliable than PDB-REDO API
"""

import requests
import json
from pathlib import Path

def query_rcsb_pdb():
    """
    Query RCSB PDB REST API for high-quality X-ray structures
    """
    print("Querying RCSB PDB REST API...")
    print("  Criteria: X-ray, resolution ≤2.5Å, protein structures")
    
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION"
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": 2.5
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "greater",
                        "value": 0
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "return_all_hits": True,
            "results_content_type": ["experimental"]
        }
    }
    
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    
    try:
        print("  Sending query...")
        response = requests.post(url, json=query, timeout=120)
        response.raise_for_status()
        
        data = response.json()
        
        if 'result_set' in data:
            pdb_ids = [hit['identifier'].lower() for hit in data['result_set']]
            print(f"  ✓ Found {len(pdb_ids)} structures")
            return pdb_ids
        else:
            print(f"  ✗ Unexpected response format")
            print(f"  Response keys: {data.keys()}")
            return None
            
    except requests.exceptions.Timeout:
        print("  ✗ Query timeout (server may be slow)")
        return None
    except Exception as e:
        print(f"  ✗ Query failed: {e}")
        return None

def main():
    print("="*60)
    print("Fetching High-Quality PDB Structures from RCSB")
    print("="*60 + "\n")
    
    # Query RCSB
    pdb_ids = query_rcsb_pdb()
    
    if not pdb_ids:
        print("\nQuery failed. No structures retrieved.")
        return
    
    # Save lists
    output_dir = Path('data/curated_lists')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Full list
    output_file = output_dir / 'rcsb_high_quality.txt'
    with open(output_file, 'w') as f:
        for pdb_id in pdb_ids:
            f.write(f"{pdb_id}\n")
    
    print(f"\n✓ Saved {len(pdb_ids)} IDs to {output_file}")
    
    # Test subsets
    test_sizes = [10, 100, 1000]
    for size in test_sizes:
        if len(pdb_ids) >= size:
            test_file = output_dir / f'rcsb_test_{size}.txt'
            with open(test_file, 'w') as f:
                for pdb_id in pdb_ids[:size]:
                    f.write(f"{pdb_id}\n")
            print(f"✓ Saved test set ({size} IDs) to {test_file}")
    
    print("\n" + "="*60)
    print("Usage Examples:")
    print("="*60)
    print("\n# Small test (10 structures):")
    print("python scripts/process_data.py \\")
    print("    --pdb_list data/curated_lists/rcsb_test_10.txt \\")
    print("    --n_structures 10 \\")
    print("    --output_dir data/small")
    print("\n# Medium dataset (1000 structures):")
    print("python scripts/process_data.py \\")
    print("    --pdb_list data/curated_lists/rcsb_test_1000.txt \\")
    print("    --n_structures 1000 \\")
    print("    --output_dir data/medium")
    print("\n# Large dataset (all structures):")
    print("python scripts/process_data.py \\")
    print("    --pdb_list data/curated_lists/rcsb_high_quality.txt \\")
    print("    --n_structures 50000 \\")
    print("    --output_dir data/full")

if __name__ == '__main__':
    main()

