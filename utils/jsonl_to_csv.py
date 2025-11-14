#!/usr/bin/env python3
"""
Convert JSONL file to CSV for Excel viewing
"""

import json
import pandas as pd
import argparse
from pathlib import Path

def jsonl_to_csv(input_file, output_file=None):
    """
    Convert JSONL file to CSV format
    
    Args:
        input_file (str): Path to input JSONL file
        output_file (str): Path to output CSV file (optional)
    """
    
    input_path = Path(input_file)
    
    if output_file is None:
        output_file = input_path.with_suffix('.csv')
    
    print(f"Converting {input_file} to {output_file}")
    
    # Read JSONL file
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    json_obj = json.loads(line)
                    data.append(json_obj)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                    continue
    
    if not data:
        print("No valid JSON objects found in the file")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Handle nested objects by converting them to strings
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)
    
    # Save to CSV
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"✅ Successfully converted {len(data)} records to {output_file}")
    print(f"📊 Columns: {list(df.columns)}")
    print(f"📏 Shape: {df.shape}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Convert JSONL to CSV for Excel viewing')
    parser.add_argument('input', help='Input JSONL file path')
    parser.add_argument('-o', '--output', help='Output CSV file path (optional)')
    
    args = parser.parse_args()
    
    try:
        output_file = jsonl_to_csv(args.input, args.output)
        print(f"\n🎉 Ready for Excel! Open: {output_file}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
