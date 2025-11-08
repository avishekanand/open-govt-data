#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
batch_fetch_eurostat.py
Batch download Eurostat datasets from eurostat_gemma3.jsonl using eurostat_fetch_one.py

Usage:
  python batch_fetch_eurostat.py --input eurostat_gemma3.jsonl --output-dir downloads --max-datasets 10
  python batch_fetch_eurostat.py --input eurostat_gemma3.jsonl --output-dir downloads --start-from 100 --max-datasets 50
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

def load_dataset_codes(jsonl_path: str) -> List[Dict]:
    """Load dataset codes and metadata from JSONL file."""
    datasets = []
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if 'code' in data:
                        datasets.append({
                            'code': data['code'],
                            'title': data.get('title', ''),
                            'line_num': line_num
                        })
                except json.JSONDecodeError as e:
                    print(f"[WARN] Skipping invalid JSON at line {line_num}: {e}")
                    continue
    except FileNotFoundError:
        print(f"[ERROR] Input file not found: {jsonl_path}")
        sys.exit(1)
    
    return datasets

def sanitize_filename(code: str, title: str = "") -> str:
    """Create a safe filename from dataset code and title."""
    # Use code as base, optionally add truncated title
    safe_code = "".join(c for c in code if c.isalnum() or c in "_-")
    if title:
        safe_title = "".join(c for c in title[:30] if c.isalnum() or c in "_- ")
        safe_title = "_".join(safe_title.split())  # normalize spaces
        if safe_title:
            return f"{safe_code}_{safe_title}.csv"
    return f"{safe_code}.csv"

def fetch_dataset(code: str, output_path: str, timeout: int = 120) -> Dict:
    """Fetch a single dataset using eurostat_fetch_one.py."""
    cmd = [
        sys.executable, 
        "eurostat_fetch_one.py",
        "--code", code,
        "--out", output_path,
        "--timeout", str(timeout)
    ]
    
    result = {
        'code': code,
        'output_path': output_path,
        'success': False,
        'error': None,
        'duration': 0,
        'file_size': 0
    }
    
    start_time = time.time()
    try:
        print(f"[INFO] Fetching {code}...")
        proc = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=timeout,
            cwd=os.getcwd()
        )
        
        result['duration'] = time.time() - start_time
        
        if proc.returncode == 0:
            if os.path.exists(output_path):
                result['file_size'] = os.path.getsize(output_path)
                result['success'] = True
                print(f"[OK] {code} → {output_path} ({result['file_size']:,} bytes, {result['duration']:.1f}s)")
            else:
                result['error'] = "Output file not created"
                print(f"[ERROR] {code}: Output file not created")
        else:
            result['error'] = f"Exit code {proc.returncode}: {proc.stderr.strip()}"
            print(f"[ERROR] {code}: {result['error']}")
            
    except subprocess.TimeoutExpired:
        result['error'] = f"Timeout after {timeout}s"
        print(f"[ERROR] {code}: Timeout after {timeout}s")
    except Exception as e:
        result['error'] = str(e)
        print(f"[ERROR] {code}: {e}")
    
    return result

def write_progress_log(log_path: str, results: List[Dict]):
    """Write progress log with results."""
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("code,success,duration_sec,file_size_bytes,error\n")
        for r in results:
            error_clean = (r['error'] or '').replace(',', ';').replace('\n', ' ')
            f.write(f"{r['code']},{r['success']},{r['duration']:.2f},{r['file_size']},{error_clean}\n")

def main():
    parser = argparse.ArgumentParser(description="Batch download Eurostat datasets")
    parser.add_argument("--input", required=True, help="Input JSONL file (e.g., eurostat_gemma3.jsonl)")
    parser.add_argument("--output-dir", default="downloads", help="Output directory for CSV files")
    parser.add_argument("--start-from", type=int, default=0, help="Start from dataset index (0-based)")
    parser.add_argument("--max-datasets", type=int, help="Maximum number of datasets to fetch")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout per dataset in seconds")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if output file already exists")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests in seconds")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load datasets
    print(f"[INFO] Loading dataset codes from {args.input}...")
    datasets = load_dataset_codes(args.input)
    print(f"[INFO] Found {len(datasets)} datasets")
    
    # Apply start/limit
    if args.start_from > 0:
        datasets = datasets[args.start_from:]
        print(f"[INFO] Starting from index {args.start_from}")
    
    if args.max_datasets:
        datasets = datasets[:args.max_datasets]
        print(f"[INFO] Limited to {len(datasets)} datasets")
    
    # Progress tracking
    results = []
    log_path = output_dir / "batch_fetch_log.csv"
    
    print(f"[INFO] Starting batch fetch of {len(datasets)} datasets...")
    print(f"[INFO] Output directory: {output_dir.absolute()}")
    print(f"[INFO] Progress log: {log_path}")
    print("-" * 80)
    
    try:
        for i, dataset in enumerate(datasets, 1):
            code = dataset['code']
            title = dataset.get('title', '')
            
            # Generate output filename
            filename = sanitize_filename(code, title)
            output_path = output_dir / filename
            
            # Skip if exists
            if args.skip_existing and output_path.exists():
                print(f"[SKIP] {i}/{len(datasets)} {code}: File already exists")
                results.append({
                    'code': code,
                    'output_path': str(output_path),
                    'success': True,
                    'error': 'skipped_existing',
                    'duration': 0,
                    'file_size': output_path.stat().st_size
                })
                continue
            
            print(f"[{i}/{len(datasets)}] Processing {code}...")
            
            # Fetch dataset
            result = fetch_dataset(code, str(output_path), args.timeout)
            results.append(result)
            
            # Write progress log
            write_progress_log(log_path, results)
            
            # Delay between requests
            if i < len(datasets) and args.delay > 0:
                time.sleep(args.delay)
                
    except KeyboardInterrupt:
        print(f"\n[INFO] Interrupted by user after {len(results)} datasets")
    
    # Final summary
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    total_size = sum(r['file_size'] for r in results if r['success'])
    total_duration = sum(r['duration'] for r in results)
    
    print("\n" + "=" * 80)
    print("BATCH FETCH SUMMARY")
    print("=" * 80)
    print(f"Total processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total data downloaded: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
    print(f"Total time: {total_duration:.1f} seconds")
    print(f"Average time per dataset: {total_duration/len(results):.1f}s" if results else "N/A")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Progress log: {log_path}")
    
    if failed > 0:
        print(f"\nFailed datasets:")
        for r in results:
            if not r['success']:
                print(f"  {r['code']}: {r['error']}")

if __name__ == "__main__":
    main()
