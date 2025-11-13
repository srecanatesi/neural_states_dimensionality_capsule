#!/usr/bin/env python3
import os
import sys
import subprocess
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'code'

def run(cmd):
    print('>>', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)

def process_single_session(session_id):
    """Process a single session: download, preprocess, and run HMM crossvalidation."""
    import time
    start_time = time.time()
    
    print(f'\n{"="*60}')
    print(f'[SESSION {session_id}] Starting processing pipeline')
    print(f'{"="*60}')
    
    # Determine directories (support both container and local paths)
    data_dir_abs = Path('/data')
    data_dir_rel = Path(__file__).resolve().parents[1] / 'data'
    results_dir_abs = Path('/results')
    results_dir_rel = Path(__file__).resolve().parents[1] / 'results'
    
    if data_dir_rel.exists() and data_dir_rel.is_dir():
        data_dir = data_dir_rel
        results_dir = results_dir_rel
    elif data_dir_abs.exists() and data_dir_abs.is_dir():
        data_dir = data_dir_abs
        results_dir = results_dir_abs
    else:
        data_dir = data_dir_rel
        results_dir = results_dir_rel
    
    print(f'[SESSION {session_id}] Data directory: {data_dir}')
    print(f'[SESSION {session_id}] Results directory: {results_dir}')
    
    # Ensure output dirs
    print(f'[SESSION {session_id}] Creating output directories...')
    (data_dir / 'sessions_nwb').mkdir(parents=True, exist_ok=True)
    (results_dir / 'sessions_preprocessed').mkdir(parents=True, exist_ok=True)
    (results_dir / 'sessions_hmm_crossval').mkdir(parents=True, exist_ok=True)
    (results_dir / 'logs').mkdir(parents=True, exist_ok=True)
    print(f'[SESSION {session_id}] [OK] Output directories ready')

    # 1) Download single session
    print(f'\n[SESSION {session_id}] STEP 1/3: Downloading NWB file...')
    print(f'[SESSION {session_id}] {"-"*50}')
    step_start = time.time()
    run(['python', str(Path(__file__).parent / 'download_single_session.py'), str(session_id)])
    step_time = time.time() - step_start
    print(f'[SESSION {session_id}] [OK] STEP 1/3 completed in {step_time:.1f}s')

    # 2) Preprocess this session
    print(f'\n[SESSION {session_id}] STEP 2/3: Preprocessing NWB data...')
    print(f'[SESSION {session_id}] {"-"*50}')
    step_start = time.time()
    run(['python', str(Path(__file__).parent / 'preprocessing_nwb.py'), str(session_id)])
    step_time = time.time() - step_start
    print(f'[SESSION {session_id}] [OK] STEP 2/3 completed in {step_time:.1f}s')

    # 3) HMM crossvalidation (fast)
    print(f'\n[SESSION {session_id}] STEP 3/3: Running HMM cross-validation...')
    print(f'[SESSION {session_id}] {"-"*50}')
    # Determine results directory for HMM script
    results_dir_abs = Path('/results')
    results_dir_rel = Path(__file__).resolve().parents[1] / 'results'
    if results_dir_rel.exists() and results_dir_rel.is_dir():
        hmm_results_dir = results_dir_rel
    elif results_dir_abs.exists() and results_dir_abs.is_dir():
        hmm_results_dir = results_dir_abs
    else:
        hmm_results_dir = results_dir_rel
    
    step_start = time.time()
    run([
        'python', str(Path(__file__).parent / 'hmm_crossvalidation_fast.py'),
        '--session-id', str(session_id),
        '--data-dir', str(hmm_results_dir / 'sessions_preprocessed'),
        '--output-dir', str(hmm_results_dir / 'sessions_hmm_crossval'),
        '--n-folds', '3', '--n-iter-xval', '5', '--n-iter-final', '50', '--n-final-fits', '1', '--tolerance', '0.1'
    ])
    step_time = time.time() - step_start
    print(f'[SESSION {session_id}] [OK] STEP 3/3 completed in {step_time:.1f}s')
    
    total_time = time.time() - start_time
    print(f'\n[SESSION {session_id}] {"="*60}')
    print(f'[SESSION {session_id}] [OK] ALL STEPS COMPLETED')
    print(f'[SESSION {session_id}] Total processing time: {total_time:.1f}s ({total_time/60:.1f} minutes)')
    print(f'[SESSION {session_id}] {"="*60}')

def main():
    # Download sessions.csv if it doesn't exist
    from download_s3 import SESSIONS_TABLE_KEY, download_file
    
    # Determine data directory (support both container /data and local data/)
    data_dir_abs = Path('/data')
    data_dir_rel = Path(__file__).resolve().parents[1] / 'data'
    
    # Prefer relative path if it exists (for local development)
    # Otherwise use absolute path (for container environments)
    if data_dir_rel.exists() and data_dir_rel.is_dir():
        data_dir = data_dir_rel
        print(f'Using relative data directory: {data_dir}')
    elif data_dir_abs.exists() and data_dir_abs.is_dir():
        data_dir = data_dir_abs
        print(f'Using absolute data directory: {data_dir}')
    else:
        # Default to relative for error messages
        data_dir = data_dir_rel
        print(f'Warning: Neither data directory found, using: {data_dir}')
    
    sessions_csv = data_dir / 'sessions.csv'
    if not sessions_csv.exists() or (sessions_csv.exists() and sessions_csv.stat().st_size == 0):
        if sessions_csv.exists() and sessions_csv.stat().st_size == 0:
            print(f'sessions.csv exists but is empty, re-downloading...')
        else:
            print(f'Downloading sessions.csv → {sessions_csv}')
        download_file(SESSIONS_TABLE_KEY, str(sessions_csv), show_progress=True, session_id=0)
    else:
        print(f'sessions.csv already exists ({sessions_csv.stat().st_size / 1024:.1f} KB), skipping download')
    
    # Load and process sessions table
    print(f'Loading sessions table from {sessions_csv}')
    df = pd.read_csv(sessions_csv)
    
    # Sort by id (ecephys_session_id)
    df = df.sort_values('id').reset_index(drop=True)
    
    # Filter by session_type == "functional_connectivity"
    df_filtered = df[df['session_type'] == 'functional_connectivity'].reset_index(drop=True)
    
    print(f'Total sessions: {len(df)}')
    print(f'Functional connectivity sessions: {len(df_filtered)}')
    
    if len(df_filtered) == 0:
        print('ERROR: No functional_connectivity sessions found', file=sys.stderr)
        sys.exit(1)
    
    # Find txt files in data folder
    txt_files = []
    print(f'Looking for txt files in: {data_dir}')
    for run_dir in ['run1', 'run2', 'run3']:
        txt_file = data_dir / run_dir / f'{run_dir[-1]}.txt'
        print(f'  Checking: {txt_file} (exists: {txt_file.exists()})')
        if txt_file.exists():
            txt_files.append((int(run_dir[-1]), txt_file))
    
    txt_files.sort(key=lambda x: x[0])  # Sort by run number
    
    if len(txt_files) == 0:
        print(f'\nNo txt files found in data/run1/, data/run2/, or data/run3/')
        print('Nothing to process. Exiting gracefully.')
        return  # Exit without error
    
    print(f'\nFound {len(txt_files)} txt file(s) to process')
    
    # Process sessions corresponding to each txt file
    # txt file 1.txt -> session index 0 (1st session)
    # txt file 2.txt -> session index 1 (2nd session)
    # txt file 3.txt -> session index 2 (3rd session)
    import time
    pipeline_start = time.time()
    
    print(f'\n{"="*60}')
    print(f'PIPELINE SUMMARY')
    print(f'{"="*60}')
    print(f'Total sessions to process: {len(txt_files)}')
    print(f'Functional connectivity sessions available: {len(df_filtered)}')
    print(f'{"="*60}\n')
    
    for idx, (run_num, txt_file) in enumerate(txt_files, 1):
        session_idx = run_num - 1  # Convert to 0-based index
        
        if session_idx >= len(df_filtered):
            print(f'\n[RUN {run_num}] WARNING: Session index {session_idx} (from {txt_file.name}) exceeds available sessions ({len(df_filtered)}). Skipping.', file=sys.stderr)
            continue
        
        session_id = str(int(df_filtered.iloc[session_idx]['id']))
        print(f'\n{"#"*60}')
        print(f'[RUN {run_num}/{len(txt_files)}] Processing session from {txt_file.name}')
        print(f'[RUN {run_num}/{len(txt_files)}] Session ID: {session_id} (index {session_idx} in filtered list)')
        print(f'{"#"*60}')
        
        process_single_session(session_id)
        
        print(f'\n[RUN {run_num}/{len(txt_files)}] [OK] Completed run {run_num}')
    
    total_pipeline_time = time.time() - pipeline_start
    print(f'\n{"="*60}')
    print('PIPELINE COMPLETE')
    print(f'{"="*60}')
    print(f'Total sessions processed: {len(txt_files)}')
    print(f'Total pipeline time: {total_pipeline_time:.1f}s ({total_pipeline_time/60:.1f} minutes)')
    print(f'Average time per session: {total_pipeline_time/len(txt_files):.1f}s')
    print(f'{"="*60}')

if __name__ == '__main__':
    main()


