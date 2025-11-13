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
    
    # Determine data directory (support both container /data and local data/)
    data_dir_abs = Path('/data')
    data_dir_rel = Path(__file__).resolve().parents[1] / 'data'
    
    # Prefer relative path if it exists (for local development)
    # Otherwise use absolute path (for container environments)
    if data_dir_rel.exists() and data_dir_rel.is_dir():
        out_dir = data_dir_rel
    elif data_dir_abs.exists() and data_dir_abs.is_dir():
        out_dir = data_dir_abs
    else:
        # Default to relative for local development
        out_dir = data_dir_rel
    
    nwb_dir = out_dir / 'sessions_nwb'
    nwb_dir.mkdir(parents=True, exist_ok=True)

    # Ensure supporting CSVs exist in data directory
    sessions_csv = out_dir / 'sessions.csv'
    if not sessions_csv.exists() or sessions_csv.stat().st_size == 0:
        if sessions_csv.exists() and sessions_csv.stat().st_size == 0:
            print(f"[Session {session_id}] sessions.csv exists but is empty, re-downloading...")
        else:
            print(f"[Session {session_id}] Fetching sessions.csv → {sessions_csv}")
        download_file(SESSIONS_TABLE_KEY, str(sessions_csv), show_progress=False, session_id=int(session_id))
    else:
        print(f"[Session {session_id}] sessions.csv already exists, skipping download")

    # Prefer local unit_table_all.csv from repository data rather than downloading
    unit_table_csv = out_dir / 'unit_table_all.csv'
    if not unit_table_csv.exists():
        # Search under code/data/**/unit_table_all.csv and copy the first match
        repo_data_root = Path(__file__).resolve().parents[1] / 'data'
        candidates = []
        if repo_data_root.exists():
            candidates = glob.glob(str(repo_data_root / '**' / 'unit_table_all.csv'), recursive=True)
        if candidates:
            src = Path(candidates[0])
            print(f"[Session {session_id}] Copying unit_table_all.csv from {src} → {unit_table_csv}")
            shutil.copy2(src, unit_table_csv)
        else:
            print(f"[Session {session_id}] WARNING: unit_table_all.csv not found; downstream preprocessing may fail.")

    key = SESSION_NWB_KEY.format(sid=session_id)
    dest = nwb_dir / f'session_{session_id}.nwb'
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[Session {session_id}] NWB file already exists ({dest.stat().st_size / (1024*1024):.1f} MB): {dest}")
        print(f"[Session {session_id}] Skipping download")
        return

    if dest.exists() and dest.stat().st_size == 0:
        print(f"[Session {session_id}] NWB file exists but is empty, re-downloading...")
        dest.unlink()  # Remove empty file

    print(f"[Session {session_id}] Downloading NWB file to {dest} ...")
    download_file(key, str(dest), show_progress=True, session_id=int(session_id))
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[Session {session_id}] ✓ Download complete ({dest.stat().st_size / (1024*1024):.1f} MB)")
    else:
        print(f"[Session {session_id}] ⚠ Warning: Download may have failed (file size: {dest.stat().st_size if dest.exists() else 0} bytes)")

if __name__ == '__main__':
    main()


