from pathlib import Path

SOURCE_REPO = "Mimic-Robotics/full_ttt_redx_stable"
TARGET_REPO = "Mimic-Robotics/full_ttt_redx_stable_fixed_labels"

BASE = Path("/home/a/ac_pate/mimic-lerobot/mimic_deployment/dataset_management/stable_red_ttt")
AUDIT_MD = BASE / "audit" / "01_human_audit_guide.md"
WORKDIR = Path("/speed-scratch/ac_pate/hf_work/full_ttt_redx_stable_fixed_labels")

REQUIRED_REMOTE_FILES = [
    "meta/tasks.parquet",
    "meta/info.json",
    "meta/episodes/chunk-000/file-000.parquet",
    "data/chunk-000/file-000.parquet",
]
