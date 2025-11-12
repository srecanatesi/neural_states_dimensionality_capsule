#!/usr/bin/env python3
"""
Minimal single-session downloader used by the capsule.
"""
import os
import io
import time
from pathlib import Path
import pandas as pd
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

BUCKET = "allen-brain-observatory"
REGION = "us-west-2"
SESSIONS_TABLE_KEY = "visual-coding-neuropixels/ecephys-cache/sessions.csv"
SESSION_NWB_KEY = "visual-coding-neuropixels/ecephys-cache/session_{sid}/session_{sid}.nwb"

def s3_client():
    return boto3.client("s3", region_name=REGION, config=Config(signature_version=UNSIGNED))

def download_file(key: str, dest_path: str, show_progress: bool = True, session_id: int = None):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    s3 = s3_client()
    try:
        head = s3.head_object(Bucket=BUCKET, Key=key)
    except ClientError as e:
        raise FileNotFoundError(f"S3 object not found: s3://{BUCKET}/{key}") from e
    size = head.get("ContentLength", None)
    bytes_read = 0
    chunk = 8 * 1024 * 1024
    try:
        with open(dest_path, "wb") as f:
            resp = s3.get_object(Bucket=BUCKET, Key=key)
            body = resp["Body"]
            t0 = time.time()
            while True:
                data = body.read(chunk)
                if not data:
                    break
                f.write(data)
                bytes_read += len(data)
                if size and show_progress and (time.time() - t0 > 2.0):
                    pct = 100.0 * bytes_read / size
                    print(f"[Session {session_id}] … {os.path.basename(dest_path)}: {pct:5.1f}%")
            if show_progress:
                elapsed = time.time() - t0
                print(f"[Session {session_id}] ✓ {os.path.basename(dest_path)} done in {elapsed:.1f}s")
    except Exception:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise
    if size and os.path.getsize(dest_path) != size:
        os.remove(dest_path)
        raise IOError("Size mismatch after download")




