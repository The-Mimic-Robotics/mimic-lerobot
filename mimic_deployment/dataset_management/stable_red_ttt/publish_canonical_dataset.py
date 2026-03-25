import argparse
import importlib.util
import json
import re
import tempfile
from pathlib import Path

import pandas as pd
from huggingface_hub import CommitOperationCopy, CommitOperationDelete, CommitOperationAdd, HfApi, hf_hub_download


SOURCE_REPO = "Mimic-Robotics/full_ttt_redx_stable"
TARGET_REPO_DEFAULT = "Mimic-Robotics/full_ttt_redx_stable_canonical_prompt"


def load_normalizer_module(script_dir: Path):
    module_path = script_dir / "instruction_normalizer.py"
    spec = importlib.util.spec_from_file_location("instruction_normalizer", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def copy_repo_server_side(api: HfApi, src_repo: str, dst_repo: str):
    src_files = api.list_repo_files(repo_id=src_repo, repo_type="dataset")
    pending = list(src_files)
    skipped_missing = []

    max_retries = max(1, len(pending) + 5)
    for _ in range(max_retries):
        ops = [CommitOperationCopy(src_repo_id=src_repo, src_path_in_repo=fp, path_in_repo=fp) for fp in pending]
        try:
            api.create_commit(
                repo_id=dst_repo,
                repo_type="dataset",
                operations=ops,
                commit_message=f"Initialize from {src_repo} via server-side copy",
            )
            return {"copied": len(pending), "skipped_missing": skipped_missing}
        except Exception as exc:
            text = str(exc)
            m = re.search(r"Cannot copy (.+?) at revision", text)
            if not m:
                raise
            missing = m.group(1)
            if missing not in pending:
                raise
            pending = [fp for fp in pending if fp != missing]
            skipped_missing.append(missing)
            print(f"Skipping missing source path during copy: {missing}")

    raise RuntimeError(f"Too many missing-file retries while server-side copying. skipped={len(skipped_missing)}")


def rewrite_tasks_and_info(api: HfApi, repo_id: str, normalizer, script_dir: Path):
    tasks_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="meta/tasks.parquet")
    tasks_df = pd.read_parquet(tasks_path)

    tf = tasks_df.reset_index()
    task_name_col = "task" if "task" in tf.columns else ("__index_level_0__" if "__index_level_0__" in tf.columns else tf.columns[0])
    task_idx_col = "task_index" if "task_index" in tf.columns else ("index" if "index" in tf.columns else tf.columns[1])

    new_rows = []
    for _, row in tf.iterrows():
        old_label = str(row[task_name_col])
        new_label = normalizer.canonical_instruction_from_legacy_task(old_label)
        new_rows.append({"task": new_label, "task_index": int(row[task_idx_col])})

    new_rows = sorted(new_rows, key=lambda x: x["task_index"])
    new_df = pd.DataFrame({"task_index": [x["task_index"] for x in new_rows]}, index=[x["task"] for x in new_rows])
    new_df.index.name = "__index_level_0__"

    with tempfile.TemporaryDirectory() as tmpd:
        tmpd = Path(tmpd)
        new_tasks_local = tmpd / "tasks.parquet"
        new_df.to_parquet(new_tasks_local)

        operations = [
            CommitOperationDelete(path_in_repo="meta/tasks.parquet"),
            CommitOperationAdd(path_in_repo="meta/tasks.parquet", path_or_fileobj=str(new_tasks_local)),
        ]

        info_candidate = "meta/info.json"
        repo_files = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
        if info_candidate in repo_files:
            info_local = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=info_candidate)
            with open(info_local, "r", encoding="utf-8") as f:
                info = json.load(f)
            if isinstance(info.get("tasks"), list):
                info["tasks"] = [x["task"] for x in new_rows]
                new_info_local = tmpd / "info.json"
                with open(new_info_local, "w", encoding="utf-8") as f:
                    json.dump(info, f, indent=2)
                operations.extend([
                    CommitOperationDelete(path_in_repo=info_candidate),
                    CommitOperationAdd(path_in_repo=info_candidate, path_or_fileobj=str(new_info_local)),
                ])

        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message="Canonicalize language prompts with uppercase cell token format",
        )

    return [x["task"] for x in new_rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", default=SOURCE_REPO)
    parser.add_argument("--target-repo", default=TARGET_REPO_DEFAULT)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    normalizer = load_normalizer_module(script_dir)

    api = HfApi()
    api.create_repo(repo_id=args.target_repo, repo_type="dataset", private=False, exist_ok=True)

    copy_stats = copy_repo_server_side(api, args.source_repo, args.target_repo)
    new_tasks = rewrite_tasks_and_info(api, args.target_repo, normalizer, script_dir)

    report_path = script_dir / "audit" / "03_publish_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"source_repo: {args.source_repo}\n")
        f.write(f"target_repo: {args.target_repo}\n")
        f.write(f"copied_files: {copy_stats['copied']}\n")
        f.write(f"skipped_missing: {len(copy_stats['skipped_missing'])}\n")
        for miss in copy_stats["skipped_missing"]:
            f.write(f"- skipped: {miss}\n")
        f.write("canonical_tasks:\n")
        for t in new_tasks:
            f.write(f"- {t}\n")
        f.write("\nvisualizer_url:\n")
        f.write(f"https://huggingface.co/spaces/lerobot/visualize_dataset?dataset={args.target_repo.replace('/', '%2F')}\n")

    print(f"Wrote {report_path}")
    print(f"Visualizer: https://huggingface.co/spaces/lerobot/visualize_dataset?dataset={args.target_repo.replace('/', '%2F')}")


if __name__ == "__main__":
    main()
