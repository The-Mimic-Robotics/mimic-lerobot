#!/usr/bin/env python3
"""
Fix dtype mismatches in converted datasets.

The source datasets have float64 in parquet files but info.json says float32.
This script fixes the parquet files to match the metadata.

Usage:
    PYTHONPATH=src python scripts/migration/fix_dtypes.py
"""

import sys
import json
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
from huggingface_hub import HfApi, snapshot_download
from lerobot.utils.constants import HF_LEROBOT_HOME


def fix_parquet_dtypes(parquet_path: Path) -> bool:
    """Fix dtypes and list types in a parquet file.

    Converts:
    - float64 to float32
    - variable-length list to fixed_size_list[15]
    """
    try:
        table = pq.read_table(parquet_path)
        schema = table.schema
        new_columns = {}
        changed = False

        for i, field in enumerate(schema):
            col_name = field.name
            col = table.column(i)

            # Check if this is an action or state column that needs fixing
            if col_name in ['action', 'observation.state']:
                needs_fix = False

                # Check if it needs dtype or list type fix
                if pa.types.is_list(field.type) or pa.types.is_fixed_size_list(field.type):
                    value_type = field.type.value_type
                    # Need fix if: float64, or variable-length list
                    if pa.types.is_float64(value_type) or (pa.types.is_list(field.type) and not pa.types.is_fixed_size_list(field.type)):
                        needs_fix = True

                if needs_fix:
                    print(f"    Converting {col_name} to fixed_size_list<float32>[15]")

                    # Convert each row to fixed-size list of float32, padding if needed
                    new_arrays = []
                    for chunk in col.chunks:
                        converted_values = []
                        for row in chunk:
                            if row is not None:
                                arr = np.array(row.as_py(), dtype=np.float32)
                                # Pad to 15D if needed (observation.state might be 12D)
                                if len(arr) < 15:
                                    arr = np.concatenate([arr, np.zeros(15 - len(arr), dtype=np.float32)])
                                converted_values.append(arr.tolist())
                            else:
                                converted_values.append(None)
                        # Use fixed_size_list with size 15
                        new_arrays.append(pa.array(converted_values, type=pa.list_(pa.float32(), 15)))

                    new_columns[col_name] = pa.chunked_array(new_arrays)
                    changed = True
                    continue

            new_columns[col_name] = col

        if changed:
            # Reconstruct table with fixed columns
            new_table = pa.table(new_columns)
            pq.write_table(new_table, parquet_path)
            return True
        return False

    except Exception as e:
        print(f"    Error fixing {parquet_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def fix_dataset(repo_id: str) -> bool:
    """Fix dtypes in a dataset and re-upload."""
    api = HfApi()

    print(f"\n{'='*60}")
    print(f"Fixing: {repo_id}")
    print('='*60)

    # Download dataset
    print("\n[Step 1/3] Downloading dataset...")
    try:
        local_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=HF_LEROBOT_HOME / repo_id,
            ignore_patterns=["*.lock", ".git*"]
        )
        local_path = Path(local_path)
    except Exception as e:
        print(f"  Error downloading: {e}")
        return False

    # Fix parquet files
    print("\n[Step 2/3] Fixing parquet dtypes...")
    data_dir = local_path / "data"
    if not data_dir.exists():
        print(f"  Error: No data directory found at {data_dir}")
        return False

    fixed_count = 0
    for pq_file in data_dir.rglob("*.parquet"):
        print(f"  Processing: {pq_file.name}")
        if fix_parquet_dtypes(pq_file):
            fixed_count += 1

    print(f"  Fixed {fixed_count} parquet files")

    # Re-upload
    print("\n[Step 3/3] Re-uploading to Hub...")
    try:
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(local_path),
            repo_type="dataset",
            ignore_patterns=[".git", ".cache", "__pycache__", "*.lock"]
        )
        print(f"  Re-uploaded: {repo_id}")
        return True
    except Exception as e:
        print(f"  Error uploading: {e}")
        return False


def main():
    # All converted mobile datasets - need fixed_size_list fix
    datasets = [
        'neryotw/mobile_bimanual_blue_block_handover_2_v30_15d',
        'neryotw/mobile_bimanual_blue_block_handover_3_v30_15d',
        'neryotw/mobile_bimanual_blue_block_handover_4_v30_15d',
        'neryotw/mobile_bimanual_blue_block_handover_5_v30_15d',
        'neryotw/mobile_bimanual_blue_block_handover_6_v30_15d',
        'neryotw/mobile_bimanual_blue_block_handover_7_v30_15d',
        'neryotw/mobile_bimanual_blue_block_handover_14_v30_15d',
        'neryotw/mobile_bimanual_blue_block_handover_15_v30_15d',
        'neryotw/mobile_bimanual_blue_block_handover_16_v30_15d',
        'neryotw/mobile_bimanual_blue_block_handover_17_v30_15d',
        'neryotw/mobile_bimanual_blue_block_handover_18_v30_15d',
        'neryotw/mobile_bimanual_blue_block_handover_19_v30_15d',
        'neryotw/mobile_bimanual_blue_block_handover_20_v30_15d',
        'neryotw/mobile_bimanual_blue_block_handover_21_v30_15d',
        'neryotw/mobile_bimanual_blue_block_handover_22_v30_15d',
        'neryotw/mobile_bimanual_blue_block_handover_23_v30_15d',
        'neryotw/mobile_bimanual_blue_block_handover_24_v30_15d',
        'neryotw/mobile_bimanual_blue_block_handover_25_v30_15d',
        'neryotw/mobile_bimanual_blue_block_handover_26_v30_15d',
        # 'neryotw/mimic_mobile_bimanual_drift_v2_v30_15d',  # Already has fixed_size_list
    ]

    print("\n" + "="*60)
    print("FIXING DTYPE MISMATCHES")
    print("="*60)
    print(f"Datasets to fix: {len(datasets)}")

    successes = []
    failures = []

    for repo_id in datasets:
        try:
            if fix_dataset(repo_id):
                successes.append(repo_id)
            else:
                failures.append(repo_id)
        except Exception as e:
            print(f"  Exception: {e}")
            failures.append(repo_id)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Fixed: {len(successes)}")
    print(f"Failed: {len(failures)}")

    if failures:
        print("\nFailed:")
        for f in failures:
            print(f"  - {f}")

    return len(failures) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
