import argparse
import ast
import json
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files


REPO_ID = "Mimic-Robotics/full_ttt_redx_stable"
BASE_DIR = Path("/home/a/ac_pate/mimic-lerobot/mimic_deployment/dataset_management/stable_red_ttt")
AUDIT_MD = BASE_DIR / "audit" / "01_human_audit_guide.md"
BLOCKS_JSON = BASE_DIR / "audit" / "00_blocks.json"
DRYRUN_REPORT = BASE_DIR / "audit" / "02_dry_run_report.txt"
CORRECTIONS_JSON = BASE_DIR / "audit" / "02_corrections_plan.json"

POSITION_TO_TASK = {
    "top_left": "pick red x handover place top left",
    "top_middle": "pick red x handover place top middle",
    "top_right": "pick red x handover place top right",
    "middle_left": "pick red x handover place middle left",
    "middle_middle": "pick red x handover place center",
    "middle_right": "pick red x handover place middle right",
    "bottom_left": "pick red x handover place bottom left",
    "bottom_middle": "pick red x handover place bottom middle",
    "bottom_right": "pick red x handover place bottom right",
}


def parse_audit_dict(md_text: str) -> dict[str, str]:
    match = re.search(r"audit\s*=\s*\{[\s\S]*?\}", md_text)
    if not match:
        raise RuntimeError("Could not find `audit = {...}` block in audit markdown.")
    expr = match.group(0).split("=", 1)[1].strip()
    audit_obj = ast.literal_eval(expr)
    if not isinstance(audit_obj, dict):
        raise RuntimeError("Parsed audit object is not a dict.")
    return {str(k): str(v) for k, v in audit_obj.items()}


def load_task_maps(repo_id: str):
    tasks_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="meta/tasks.parquet")
    tasks_df = pd.read_parquet(tasks_path)
    tf = tasks_df.reset_index()
    task_name_col = "task" if "task" in tf.columns else ("__index_level_0__" if "__index_level_0__" in tf.columns else tf.columns[0])
    task_idx_col = "task_index" if "task_index" in tf.columns else ("index" if "index" in tf.columns else tf.columns[1])
    idx_to_task = {int(row[task_idx_col]): str(row[task_name_col]) for _, row in tf.iterrows()}
    task_to_idx = {v: k for k, v in idx_to_task.items()}
    return idx_to_task, task_to_idx


def normalize_episode_tasks_col(series: pd.Series) -> pd.Series:
    def norm(value):
        v = value
        if hasattr(v, "tolist"):
            v = v.tolist()
        if isinstance(v, (list, tuple)) and len(v) > 0:
            v = v[0]
        return int(v)

    return series.map(norm)


def build_corrections(blocks: list[dict], audit_dict: dict[str, str], task_to_idx: dict[str, int]) -> list[dict]:
    corrections = []
    for block in blocks:
        key = f"ep_{block['start']}_{block['end']}"
        if key not in audit_dict:
            raise RuntimeError(f"Missing key in audit dict: {key}")
        pos = audit_dict[key].strip()
        if pos not in POSITION_TO_TASK:
            raise RuntimeError(f"Invalid position '{pos}' for {key}")
        target_label = POSITION_TO_TASK[pos]
        if target_label not in task_to_idx:
            raise RuntimeError(f"Target label '{target_label}' not present in meta/tasks.parquet")
        target_idx = int(task_to_idx[target_label])
        current_idx = int(block["task_index"])
        corrections.append(
            {
                "key": key,
                "start": int(block["start"]),
                "end": int(block["end"]),
                "current_idx": current_idx,
                "current_label": block["task"],
                "target_pos": pos,
                "target_label": target_label,
                "target_idx": target_idx,
                "changed": target_idx != current_idx,
            }
        )
    return corrections


def apply_mapping_to_frame_df(df: pd.DataFrame, episode_to_target_idx: dict[int, int]) -> tuple[pd.DataFrame, int]:
    out = df.copy()
    mask = out["episode_index"].isin(list(episode_to_target_idx.keys()))
    if not mask.any():
        return out, 0

    expected = out.loc[mask, "episode_index"].map(episode_to_target_idx).astype(int)
    current = out.loc[mask, "task_index"].astype(int)
    changes = int((current != expected).sum())
    out.loc[mask, "task_index"] = expected.values
    return out, changes


def apply_mapping_to_episodes_df(df: pd.DataFrame, episode_to_target_idx: dict[int, int]) -> tuple[pd.DataFrame, int]:
    out = df.copy()
    if "tasks" not in out.columns:
        raise RuntimeError("Expected 'tasks' column in meta/episodes parquet")

    current_idx = normalize_episode_tasks_col(out["tasks"]) 
    mask = out["episode_index"].isin(list(episode_to_target_idx.keys()))
    expected = out.loc[mask, "episode_index"].map(episode_to_target_idx).astype(int)
    changes = int((current_idx.loc[mask].astype(int) != expected).sum())

    out.loc[mask, "tasks"] = expected.map(lambda x: [int(x)])
    return out, changes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Actually write patched parquet files")
    parser.add_argument("--repo-id", default=REPO_ID)
    args = parser.parse_args()

    audit_md = AUDIT_MD.read_text(encoding="utf-8")
    audit_dict = parse_audit_dict(audit_md)
    blocks_data = json.loads(BLOCKS_JSON.read_text(encoding="utf-8"))
    blocks = blocks_data["blocks"]

    idx_to_task, task_to_idx = load_task_maps(args.repo_id)
    corrections = build_corrections(blocks, audit_dict, task_to_idx)

    changed_blocks = [c for c in corrections if c["changed"]]
    target_episodes = []
    for c in changed_blocks:
        target_episodes.extend(range(c["start"], c["end"] + 1))
    episode_to_target_idx = {}
    for c in changed_blocks:
        for ep in range(c["start"], c["end"] + 1):
            episode_to_target_idx[ep] = c["target_idx"]

    files = list_repo_files(repo_id=args.repo_id, repo_type="dataset")
    data_files = sorted([f for f in files if f.startswith("data/") and f.endswith(".parquet")])
    ep_meta_file = "meta/episodes/chunk-000/file-000.parquet"

    dry_lines = []
    dry_lines.append(f"Repo: {args.repo_id}")
    dry_lines.append(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    dry_lines.append("")
    dry_lines.append("Per-block mapping:")
    for c in corrections:
        marker = "CHANGE" if c["changed"] else "OK"
        dry_lines.append(
            f"- {c['key']}: {marker} | {c['current_idx']} ({c['current_label']}) -> {c['target_idx']} ({c['target_label']})"
        )
    dry_lines.append("")
    dry_lines.append(f"Changed blocks: {len(changed_blocks)} / {len(corrections)}")
    dry_lines.append(f"Target episodes: {len(set(target_episodes))}")

    total_frame_rows_changed = 0
    touched_data_files = 0

    for f in data_files:
        local = hf_hub_download(repo_id=args.repo_id, repo_type="dataset", filename=f)
        df = pd.read_parquet(local)
        if "episode_index" not in df.columns or "task_index" not in df.columns:
            raise RuntimeError(f"File {f} missing required columns (episode_index/task_index)")
        _, changed = apply_mapping_to_frame_df(df, episode_to_target_idx)
        if changed > 0:
            touched_data_files += 1
            total_frame_rows_changed += changed
        dry_lines.append(f"- data file {f}: rows_to_change={changed}")

        if args.apply and changed > 0:
            patched_df, _ = apply_mapping_to_frame_df(df, episode_to_target_idx)
            patched_df.to_parquet(local, index=False)

    local_ep = hf_hub_download(repo_id=args.repo_id, repo_type="dataset", filename=ep_meta_file)
    ep_df = pd.read_parquet(local_ep)
    _, ep_changes = apply_mapping_to_episodes_df(ep_df, episode_to_target_idx)
    dry_lines.append(f"- episodes meta {ep_meta_file}: rows_to_change={ep_changes}")

    if args.apply and ep_changes > 0:
        patched_ep, _ = apply_mapping_to_episodes_df(ep_df, episode_to_target_idx)
        patched_ep.to_parquet(local_ep, index=False)

    dry_lines.append("")
    dry_lines.append(f"Summary: frame_rows_to_change={total_frame_rows_changed}, episode_rows_to_change={ep_changes}, data_files_touched={touched_data_files}")

    DRYRUN_REPORT.write_text("\n".join(dry_lines) + "\n", encoding="utf-8")

    output_plan = {
        "repo": args.repo_id,
        "mode": "apply" if args.apply else "dry-run",
        "changed_blocks": changed_blocks,
        "summary": {
            "changed_blocks": len(changed_blocks),
            "total_blocks": len(corrections),
            "target_episodes": len(set(target_episodes)),
            "frame_rows_to_change": total_frame_rows_changed,
            "episode_rows_to_change": ep_changes,
            "data_files_touched": touched_data_files,
        },
    }
    CORRECTIONS_JSON.write_text(json.dumps(output_plan, indent=2), encoding="utf-8")

    print(f"Wrote {DRYRUN_REPORT}")
    print(f"Wrote {CORRECTIONS_JSON}")


if __name__ == "__main__":
    main()
