"""
PDB-REDO and RCSB PDB downloader with diverse structure selection
"""
import requests
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import random


class PDBRedoDownloader:
    """Download structures from PDB-REDO database with RCSB fallback"""
    
    def __init__(self, output_dir="data/pdb_redo"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pdb_redo_url = "https://pdb-redo.eu/db"
        self.rcsb_url = "https://files.rcsb.org/download"
    
    def download_structure(self, pdb_id, use_fallback=True):
        """Download a single structure with fallback to RCSB"""
        pdb_id_lower = pdb_id.lower()
        pdb_id_upper = pdb_id.upper()
        
        output_file = self.output_dir / f"{pdb_id_lower}_redo.pdb"
        
        if output_file.exists():
            # print(f"Skipping {pdb_id} (already downloaded)")  # Commented to reduce spam
            return output_file
        
        # Try PDB-REDO first
        try:
            url = f"{self.pdb_redo_url}/{pdb_id_lower}/{pdb_id_lower}_final.pdb"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(output_file, 'w') as f:
                f.write(response.text)
            
            print(f"✓ {pdb_id} from PDB-REDO")
            return output_file
            
        except requests.exceptions.RequestException as e:
            if not use_fallback:
                print(f"✗ PDB-REDO failed for {pdb_id}")
                return None
            
            # Fallback to regular RCSB PDB
            try:
                url = f"{self.rcsb_url}/{pdb_id_upper}.pdb"
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # Save with different filename to indicate it's from RCSB
                output_file = self.output_dir / f"{pdb_id_lower}.pdb"
                with open(output_file, 'w') as f:
                    f.write(response.text)
                
                print(f"✓ {pdb_id} from RCSB (fallback)")
                return output_file
                
            except requests.exceptions.RequestException as e2:
                print(f"✗ Both sources failed for {pdb_id}")
                return None
    
    def download_batch(self, pdb_ids, max_workers=5):
        """Download multiple structures with rate limiting"""
        downloaded = []
        failed = []
        
        print(f"\nDownloading {len(pdb_ids)} structures...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.download_structure, pdb_id): pdb_id 
                for pdb_id in pdb_ids
            }
            
            for future in as_completed(futures):
                pdb_id = futures[future]
                try:
                    result = future.result()
                    if result:
                        downloaded.append(result)
                    else:
                        failed.append(pdb_id)
                except Exception as e:
                    print(f"✗ Error processing {pdb_id}: {e}")
                    failed.append(pdb_id)
                
                time.sleep(0.1)  # Rate limiting
        
        print(f"\nDownload summary:")
        print(f"  Success: {len(downloaded)}")
        print(f"  Failed: {len(failed)}")
        
        return downloaded, failed
    
    def download_structures(self, pdb_ids, max_workers=5):
        """Alias for download_batch (for compatibility)"""
        downloaded, failed = self.download_batch(pdb_ids, max_workers)
        return downloaded


class RSCBDownloader:
    """Download structures directly from RCSB PDB"""
    
    def __init__(self, output_dir="data/rcsb_pdb"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://files.rcsb.org/download"
    
    def download_structure(self, pdb_id):
        """Download a single structure from RCSB PDB"""
        pdb_id_lower = pdb_id.lower()
        pdb_id_upper = pdb_id.upper()
        
        output_file = self.output_dir / f"{pdb_id_lower}.pdb"
        
        if output_file.exists():
            # print(f"Skipping {pdb_id} (already downloaded)")  # Commented to reduce spam
            return output_file
        
        try:
            url = f"{self.base_url}/{pdb_id_upper}.pdb"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(output_file, 'w') as f:
                f.write(response.text)
            
            print(f"✓ {pdb_id}")
            return output_file
            
        except requests.exceptions.RequestException as e:
            print(f"✗ {pdb_id}: {e}")
            return None
    
    def download_batch(self, pdb_ids, max_workers=5):
        """Download multiple structures with parallel workers"""
        downloaded = []
        failed = []
        
        print(f"\nDownloading {len(pdb_ids)} structures from RCSB...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.download_structure, pdb_id): pdb_id 
                for pdb_id in pdb_ids
            }
            
            for future in as_completed(futures):
                pdb_id = futures[future]
                try:
                    result = future.result()
                    if result:
                        downloaded.append(result)
                    else:
                        failed.append(pdb_id)
                except Exception as e:
                    print(f"✗ Error processing {pdb_id}: {e}")
                    failed.append(pdb_id)
                
                time.sleep(0.1)  # Rate limiting
        
        print(f"\nDownload summary:")
        print(f"  Success: {len(downloaded)}")
        print(f"  Failed: {len(failed)}")
        
        return downloaded, failed
    
    def download_structures(self, pdb_ids, max_workers=5):
        """Alias for download_batch (for compatibility)"""
        downloaded, failed = self.download_batch(pdb_ids, max_workers)
        return downloaded


def get_diverse_from_rcsb(target_count=3000):
    """
    Fetch diverse structures using RCSB sequence clustering
    
    Returns:
        List of diverse PDB IDs
    """
    print(f"Fetching RCSB 40% sequence identity clusters...")
    
    try:
        # Get cluster representatives (40% seq identity = very diverse)
        url = "https://cdn.rcsb.org/resources/sequence/clusters/bc-40.out"
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        
        # Parse: first entry in each line is cluster representative
        cluster_reps = []
        for line in response.text.split('\n'):
            if line.strip():
                entries = line.split()
                if entries:
                    pdb_id = entries[0].lower()[:4]
                    cluster_reps.append(pdb_id)
        
        print(f"  Found {len(cluster_reps)} cluster representatives")
        
        # Filter for high quality via RCSB query
        print(f"  Filtering for X-ray, resolution ≤2.5Å...")
        
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
                    }
                ]
            },
            "return_type": "entry",
            "request_options": {"return_all_hits": True}
        }
        
        rcsb_url = "https://search.rcsb.org/rcsbsearch/v2/query"
        resp = requests.post(rcsb_url, json=query, timeout=120)
        
        if resp.status_code == 200:
            high_quality = set(h['identifier'].lower() for h in resp.json()['result_set'])
            
            # Intersect: diverse AND high-quality
            final_ids = [pid for pid in cluster_reps if pid in high_quality]
            
            print(f"  Final diverse + high-quality: {len(final_ids)} structures")
            
            # Shuffle and limit
            random.seed(42)
            random.shuffle(final_ids)
            
            return final_ids[:target_count]
        else:
            print(f"  RCSB query failed, using cluster reps directly")
            return cluster_reps[:target_count]
            
    except Exception as e:
        print(f"  Error fetching diverse structures: {e}")
        print(f"  Falling back to simple query...")
        return get_diverse_from_simple_query(target_count)


def get_diverse_from_simple_query(target_count=3000):
    """
    Fallback: simple RCSB query for diverse sizes
    """
    print("Using simple size-based diversity query...")
    
    # Query for proteins of varying sizes (promotes diversity)
    size_ranges = [
        (50, 150),    # Small
        (150, 250),   # Medium-small
        (250, 400),   # Medium
        (400, 600),   # Medium-large
        (600, 1000),  # Large
    ]
    
    all_ids = []
    per_range = target_count // len(size_ranges)
    
    for min_size, max_size in size_ranges:
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
                            "attribute": "rcsb_entry_info.deposited_polymer_monomer_count",
                            "operator": "range",
                            "value": {"from": min_size, "to": max_size}
                        }
                    }
                ]
            },
            "return_type": "entry",
            "request_options": {"return_all_hits": True}
        }
        
        try:
            url = "https://search.rcsb.org/rcsbsearch/v2/query"
            resp = requests.post(url, json=query, timeout=120)
            
            if resp.status_code == 200:
                ids = [h['identifier'].lower() for h in resp.json()['result_set']]
                
                # Sample from this size range
                random.seed(42)
                sampled = random.sample(ids, min(per_range, len(ids)))
                all_ids.extend(sampled)
                
                print(f"  Size {min_size}-{max_size}: sampled {len(sampled)} structures")
        except:
            continue
    
    return all_ids


# For backward compatibility
def get_curated_pdb_list(list_file=None, use_diverse=True):
    """
    Load or generate a curated list of PDB IDs
    
    Args:
        list_file: Path to text file with PDB IDs (one per line)
        use_diverse: If True and no list_file, fetch diverse set from RCSB
    """
    if list_file and Path(list_file).exists():
        with open(list_file) as f:
            pdb_ids = [line.strip().lower() for line in f if line.strip()]
        print(f"Loaded {len(pdb_ids)} PDB IDs from {list_file}")
        return pdb_ids
    
    if use_diverse:
        print("Fetching diverse, high-quality structures from RCSB...")
        return get_diverse_from_rcsb()
    
    # Fallback: small test set
    print("Using fallback test set")
    return ['1ubq', '2lyz', '1aki', '1bpi', '1ctf']