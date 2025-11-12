#!/usr/bin/env python3
import os
import sys
import glob
import shutil
from pathlib import Path

from download_s3 import SESSION_NWB_KEY, SESSIONS_TABLE_KEY, download_file  # local helper

def main():
    sid = os.environ.get('SESSION_ID') or (len(sys.argv) > 1 and sys.argv[1])
    if not sid:
        print('Usage: SESSION_ID=<id> download_single_session.py', file=sys.stderr)
        sys.exit(1)

    session_id = str(int(sid))
    out_dir = Path('/data')
    nwb_dir = out_dir / 'sessions_nwb'
    nwb_dir.mkdir(parents=True, exist_ok=True)

    # Ensure supporting CSVs exist in /data
    sessions_csv = out_dir / 'sessions.csv'
    if not sessions_csv.exists():
        print(f"[Session {session_id}] Fetching sessions.csv → {sessions_csv}")
        download_file(SESSIONS_TABLE_KEY, str(sessions_csv), show_progress=False, session_id=int(session_id))

    # Prefer local unit_table_all.csv from repository data rather than downloading
    unit_table_csv = out_dir / 'unit_table_all.csv'
    if not unit_table_csv.exists():
        # Search under /code/data/**/unit_table_all.csv and copy the first match
        repo_data_root = Path('/code') / 'data'
        candidates = []
        if repo_data_root.exists():
            candidates = glob.glob(str(repo_data_root / '**' / 'unit_table_all.csv'), recursive=True)
        if candidates:
            src = Path(candidates[0])
            print(f"[Session {session_id}] Copying unit_table_all.csv from {src} → {unit_table_csv}")
            shutil.copy2(src, unit_table_csv)
        else:
            print(f"[Session {session_id}] WARNING: unit_table_all.csv not found under /code/data/**; downstream preprocessing may fail.")

    key = SESSION_NWB_KEY.format(sid=session_id)
    dest = nwb_dir / f'session_{session_id}.nwb'
    if dest.exists():
        print(f"[Session {session_id}] Already exists: {dest}")
        return

    print(f"[Session {session_id}] Downloading to {dest} ...")
    download_file(key, str(dest), show_progress=True, session_id=int(session_id))
    print(f"[Session {session_id}] Done.")

if __name__ == '__main__':
    main()


