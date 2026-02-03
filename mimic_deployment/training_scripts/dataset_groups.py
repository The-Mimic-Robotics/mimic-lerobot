#!/usr/bin/env python3
"""
Dataset Groups Resolver for Mimic Robot Training
Resolves dataset group names to full dataset repository IDs
"""

import argparse
import sys
from pathlib import Path

import yaml


def load_dataset_groups(yaml_path: Path = None) -> dict:
    """Load dataset groups from YAML file."""
    if yaml_path is None:
        yaml_path = Path(__file__).parent / "dataset_groups.yaml"
    
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    
    return config.get("groups", {})


def resolve_groups(group_names: list[str], yaml_path: Path = None) -> list[str]:
    """
    Resolve one or more group names to a list of dataset repository IDs.
    
    Args:
        group_names: List of group names (e.g., ['all_datasets', 'high_quality'])
        yaml_path: Optional path to dataset_groups.yaml
    
    Returns:
        List of unique dataset repository IDs
    """
    groups = load_dataset_groups(yaml_path)
    
    datasets = []
    for group_name in group_names:
        if group_name not in groups:
            available = ", ".join(sorted(groups.keys()))
            raise ValueError(
                f"Unknown dataset group '{group_name}'. "
                f"Available groups: {available}"
            )
        datasets.extend(groups[group_name])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_datasets = []
    for dataset in datasets:
        if dataset not in seen:
            seen.add(dataset)
            unique_datasets.append(dataset)
    
    return unique_datasets


def format_for_bash(datasets: list[str]) -> str:
    """Format dataset list for bash script (Python list format)."""
    dataset_str = ",".join(datasets)
    return f"[{dataset_str}]"


def format_for_cli(datasets: list[str]) -> str:
    """Format dataset list for CLI argument (Python list format)."""
    dataset_str = ",".join(datasets)
    return f"[{dataset_str}]"


def main():
    parser = argparse.ArgumentParser(
        description="Resolve dataset groups to repository IDs"
    )
    parser.add_argument(
        "groups",
        nargs="*",
        help="Dataset group name(s) to resolve"
    )
    parser.add_argument(
        "--format",
        choices=["bash", "cli", "list"],
        default="bash",
        help="Output format (bash: quoted list, cli: unquoted list, list: newline-separated)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to dataset_groups.yaml (default: ./dataset_groups.yaml)"
    )
    parser.add_argument(
        "--list-groups",
        action="store_true",
        help="List all available groups and exit"
    )
    
    args = parser.parse_args()
    
    try:
        groups = load_dataset_groups(args.config)
        
        if args.list_groups:
            print("Available dataset groups:")
            for group_name in sorted(groups.keys()):
                count = len(groups[group_name])
                print(f"  {group_name:20s} ({count:2d} datasets)")
            return 0
        
        if not args.groups:
            print("Error: dataset group name(s) required", file=sys.stderr)
            return 1
        
        datasets = resolve_groups(args.groups, args.config)
        
        if args.format == "bash":
            print(format_for_bash(datasets))
        elif args.format == "cli":
            print(format_for_cli(datasets))
        else:  # list
            for dataset in datasets:
                print(dataset)
        
        return 0
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
