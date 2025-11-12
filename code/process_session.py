#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'code'

def run(cmd):
    print('>>', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)

def main():
    session_id = os.environ.get('SESSION_ID') or (len(sys.argv) > 1 and sys.argv[1])
    if not session_id:
        print('Missing SESSION_ID', file=sys.stderr)
        sys.exit(1)

    # Ensure output dirs
    Path('/data/sessions_nwb').mkdir(parents=True, exist_ok=True)
    Path('/results/sessions_preprocessed').mkdir(parents=True, exist_ok=True)
    Path('/results/sessions_hmm_crossval').mkdir(parents=True, exist_ok=True)
    Path('/results/logs').mkdir(parents=True, exist_ok=True)

    # 1) Download single session
    run(['python', str(Path(__file__).parent / 'download_single_session.py'), str(session_id)])

    # 2) Preprocess this session; preprocessing_nwb.py expects a positional index.
    # Map session_id to index within the sorted list of NWB files.
    nwb_dir = Path('/data/sessions_nwb')
    files = sorted(nwb_dir.glob('session_*.nwb'))
    target = nwb_dir / f'session_{session_id}.nwb'
    if target not in files:
        raise FileNotFoundError(f'NWB not found: {target}')
    idx = files.index(target)
    run(['python', str(Path(__file__).parent / 'preprocessing_nwb.py'), str(session_id)])

    # 3) HMM crossvalidation (fast)
    run([
        'python', str(Path(__file__).parent / 'hmm_crossvalidation_fast.py'),
        '--session-id', str(session_id),
        '--data-dir', '/results/sessions_preprocessed',
        '--output-dir', '/results/sessions_hmm_crossval',
        '--n-folds', '3', '--n-iter-xval', '5', '--n-iter-final', '50', '--n-final-fits', '1', '--tolerance', '0.1'
    ])

if __name__ == '__main__':
    main()


