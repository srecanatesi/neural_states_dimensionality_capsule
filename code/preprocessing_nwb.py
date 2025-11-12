#!/usr/bin/env python3
"""
Code Ocean capsule version of preprocessing_nwb.py
Processes a single session by session_id (not index)
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from pynwb import NWBHDF5IO
import warnings
warnings.simplefilter(action='ignore')

# Import preprocessing functions from parent directory's preprocessing_nwb.py
# For Code Ocean, we'll include the essential functions inline
# Set up paths for Code Ocean
datafolder = Path('/data')
savefolder = Path('/data/sessions_preprocessed')
nwb_folder = datafolder / 'sessions_nwb'
unit_table_file = datafolder / 'unit_table_all.csv'
sessions_file = datafolder / 'sessions.csv'

# Create output directory
savefolder.mkdir(parents=True, exist_ok=True)

# Import the main preprocessing function - we'll need to adapt it
# For now, let's import from the parent code directory if available
# Otherwise, we'll need to include the full preprocessing logic

def main():
    if len(sys.argv) < 2:
        print("Usage: preprocessing_nwb.py <SESSION_ID>", file=sys.stderr)
        sys.exit(2)

    session_id = str(sys.argv[1])
    nwb_file_path = nwb_folder / f'session_{session_id}.nwb'
    
    if not nwb_file_path.exists():
        print(f"ERROR: Missing NWB file: {nwb_file_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing session {session_id} from {nwb_file_path}")
    
    # Try to import and use the original preprocessing script
    # For Code Ocean, we need to call it with the session index
    # First, find the index of this session in the sorted list
    files = sorted(nwb_folder.glob('session_*.nwb'))
    try:
        idx = files.index(nwb_file_path)
    except ValueError:
        print(f"ERROR: Could not find {nwb_file_path} in file list", file=sys.stderr)
        sys.exit(1)
    
    # Call the original preprocessing script with the session index
    # The original script expects an index, not a session_id
    # In Code Ocean, try to find the original preprocessing script
    # It might be at /code/code/preprocessing_nwb.py (if repo root is mounted)
    # or we need to include it in the capsule
    import subprocess
    
    # Try multiple possible locations for the preprocessing script
    possible_paths = [
        Path('/code') / 'code' / 'preprocessing_nwb.py',  # If repo root is mounted
        Path('/code') / 'preprocessing_nwb.py',  # If code/ is mounted directly
    ]
    
    preprocessing_script = None
    for path in possible_paths:
        if path.exists():
            preprocessing_script = path
            break
    
    if preprocessing_script is None:
        print(f"ERROR: Could not find preprocessing script", file=sys.stderr)
        print(f"Tried: {possible_paths}", file=sys.stderr)
        print("This capsule requires preprocessing_nwb.py to be available.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Using preprocessing script: {preprocessing_script}")
    result = subprocess.run(
        ['python', str(preprocessing_script), str(idx)],
        cwd='/',
        capture_output=False
    )
    
    if result.returncode != 0:
        print(f"ERROR: Preprocessing failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)
    
    # Move output from /data to /results for Code Ocean
    src = savefolder / f'df_{session_id}.pkl'
    dst_dir = Path('/results/sessions_preprocessed')
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    if src.exists():
        import shutil
        shutil.copy2(src, dst_dir / src.name)
        print(f"Saved: {dst_dir / src.name}")
    else:
        print(f"WARNING: Expected output not found: {src}", file=sys.stderr)

if __name__ == '__main__':
    main()
