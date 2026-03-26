import json
import os
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, snapshot_download
from tqdm.auto import tqdm

for env_var in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "ARROW_NUM_THREADS",
]:
    os.environ.setdefault(env_var, "1")

SOURCE_REPO = "Mimic-Robotics/full_ttt_redx_stable_fixed_labels"
TARGET_REPO = "Mimic-Robotics/full_ttt_redx_stable_fixed_short_labels"
WORKDIR = Path("/speed-scratch/ac_pate/hf_work/full_ttt_redx_stable_fixed_short_labels")
BASE = Path("/home/a/ac_pate/mimic-lerobot/mimic_deployment/dataset_management/stable_red_ttt")
REPORT_PATH = BASE / "audit" / "09_short_labels_report.json"

SHORT_FROM_SLOT = {
    "BOTTOM_LEFT": "pick red x handover place bottom left",
    "BOTTOM_MIDDLE": "pick red x handover place bottom middle",
    "BOTTOM_RIGHT": "pick red x handover place bottom right",
    "MIDDLE_LEFT": "pick red x handover place middle left",
    "MIDDLE_MIDDLE": "pick red x handover place center",
    "MIDDLE_RIGHT": "pick red x handover place middle right",
    "TOP_LEFT": "pick red x handover place top left",
    "TOP_MIDDLE": "pick red x handover place top middle",
    "TOP_RIGHT": "pick red x handover place top right",
}

SHORT_LABELS = set(SHORT_FROM_SLOT.values())


def to_short_label(task_text: str) -> str:
    text = re.sub(r"\s+", " ", task_text.strip())
    lowered = text.lower()
    if lowered in SHORT_LABELS:
        return lowered

    slot_match = re.search(r"<([A-Z_]+)>", text)
    if slot_match:
        slot = slot_match.group(1)
        if slot in SHORT_FROM_SLOT:
            return SHORT_FROM_SLOT[slot]

    raise RuntimeError(f"Unrecognized task prompt format: {task_text}")


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if rel.parts and rel.parts[0] == ".cache":
            continue
        yield path, rel.as_posix()


def patch_task_texts() -> dict:
    tasks_path = WORKDIR / "meta" / "tasks.parquet"
    info_path = WORKDIR / "meta" / "info.json"

    tasks_df = pd.read_parquet(tasks_path).reset_index()
    name_col = "task" if "task" in tasks_df.columns else ("__index_level_0__" if "__index_level_0__" in tasks_df.columns else tasks_df.columns[0])
    idx_col = "task_index" if "task_index" in tasks_df.columns else ("index" if "index" in tasks_df.columns else tasks_df.columns[1])

    rewritten = []
    changed = 0
    for _, row in tasks_df.iterrows():
        old_text = str(row[name_col])
        new_text = to_short_label(old_text)
        if old_text != new_text:
            changed += 1
        rewritten.append((int(row[idx_col]), new_text))

    rewritten = sorted(rewritten, key=lambda x: x[0])
    new_tasks_df = pd.DataFrame({"task_index": [task_index for task_index, _ in rewritten]}, index=[task_text for _, task_text in rewritten])
    new_tasks_df.index.name = "__index_level_0__"
    new_tasks_df.to_parquet(tasks_path)

    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["tasks"] = [task_text for _, task_text in rewritten]
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    return {
        "task_rows_total": len(rewritten),
        "task_rows_changed": changed,
        "tasks_preview": [{"task_index": i, "task": t} for i, t in rewritten],
        "codebase_version": info.get("codebase_version"),
    }


def main():
    api = HfApi()

    WORKDIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading source dataset to {WORKDIR} ...")
    snapshot_download(
        repo_id=SOURCE_REPO,
        repo_type="dataset",
        local_dir=WORKDIR,
        revision="main",
    )

    patch_summary = patch_task_texts()
    print(
        f"Patched tasks.parquet/info.json: {patch_summary['task_rows_changed']}/{patch_summary['task_rows_total']} rows changed"
    )

    print(f"Creating/updating target dataset repo {TARGET_REPO} ...")
    api.create_repo(repo_id=TARGET_REPO, repo_type="dataset", private=False, exist_ok=True)

    existing = set(api.list_repo_files(repo_id=TARGET_REPO, repo_type="dataset"))
    all_files = list(iter_files(WORKDIR))

    uploaded = 0
    skipped = 0
    failed = []
    for local_path, path_in_repo in tqdm(all_files, desc="Uploading files", unit="file"):
        if path_in_repo in existing:
            skipped += 1
            continue
        try:
            api.upload_file(
                repo_id=TARGET_REPO,
                repo_type="dataset",
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                commit_message=f"Upload {path_in_repo}",
            )
            uploaded += 1
        except Exception as exc:
            failed.append({"path": path_in_repo, "error": str(exc)})

    codebase_version = patch_summary.get("codebase_version")
    tags_before = [t.name for t in api.list_repo_refs(TARGET_REPO, repo_type="dataset").tags]
    if codebase_version and codebase_version not in tags_before:
        api.create_tag(
            repo_id=TARGET_REPO,
            repo_type="dataset",
            tag=codebase_version,
            revision="main",
        )
    tags_after = [t.name for t in api.list_repo_refs(TARGET_REPO, repo_type="dataset").tags]

    report = {
        "source_repo": SOURCE_REPO,
        "target_repo": TARGET_REPO,
        "workdir": str(WORKDIR),
        "patch_summary": patch_summary,
        "upload": {
            "total_local_files": len(all_files),
            "uploaded": uploaded,
            "skipped_existing": skipped,
            "failed": failed,
        },
        "tags": {
            "before": tags_before,
            "after": tags_after,
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"repo={TARGET_REPO}")
    print(f"uploaded={uploaded} skipped={skipped} failed={len(failed)}")
    print(f"tags_after={tags_after}")
    print(f"report={REPORT_PATH}")


if __name__ == "__main__":
    main()
