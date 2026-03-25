import ast
import json
import re
from pathlib import Path

import pandas as pd

BASE = Path("/home/a/ac_pate/mimic-lerobot/mimic_deployment/dataset_management/stable_red_ttt")
WORKDIR = Path("/speed-scratch/ac_pate/hf_work/full_ttt_redx_stable_fixed_labels")
AUDIT_MD = BASE / "audit" / "01_human_audit_guide.md"
OUT_JSON = BASE / "audit" / "07_fix_proof.json"
OUT_TXT = BASE / "audit" / "07_fix_proof.txt"


def parse_audit_dict(md_text: str) -> dict[str, str]:
    match = re.search(r"audit\s*=\s*\{[\s\S]*?\}", md_text)
    if not match:
        raise RuntimeError("Could not find audit dict in markdown")
    obj = ast.literal_eval(match.group(0).split("=", 1)[1].strip())
    return {str(k): str(v) for k, v in obj.items()}


def normalize_tasks_col(series: pd.Series) -> pd.Series:
    def to_int(value):
        v = value
        if hasattr(v, "tolist"):
            v = v.tolist()
        if isinstance(v, (list, tuple)) and len(v) > 0:
            v = v[0]
        return int(v)

    return series.map(to_int)


def infer_position_from_prompt(prompt: str) -> str | None:
    m = re.search(r"<([A-Z_]+)>", prompt)
    if not m:
        return None
    pos = m.group(1).strip().lower()
    valid = {
        "top_left", "top_middle", "top_right",
        "middle_left", "middle_middle", "middle_right",
        "bottom_left", "bottom_middle", "bottom_right",
    }
    return pos if pos in valid else None


def main():
    audit = parse_audit_dict(AUDIT_MD.read_text(encoding="utf-8"))

    tasks_df = pd.read_parquet(WORKDIR / "meta" / "tasks.parquet").reset_index()
    t_name = "task" if "task" in tasks_df.columns else ("__index_level_0__" if "__index_level_0__" in tasks_df.columns else tasks_df.columns[0])
    t_idx = "task_index" if "task_index" in tasks_df.columns else ("index" if "index" in tasks_df.columns else tasks_df.columns[1])

    task_index_to_prompt = {int(r[t_idx]): str(r[t_name]) for _, r in tasks_df.iterrows()}
    position_to_task_index = {}
    prompt_format_issues = []

    prompt_re = re.compile(r"^Pick the red X, hand over, then place in cell <[A-Z_]+> of the 3x3 board\.$")
    for idx, prompt in sorted(task_index_to_prompt.items()):
        if not prompt_re.match(prompt):
            prompt_format_issues.append({"task_index": idx, "prompt": prompt})
        pos = infer_position_from_prompt(prompt)
        if pos is None:
            prompt_format_issues.append({"task_index": idx, "prompt": prompt, "reason": "missing/invalid <SLOT>"})
        else:
            position_to_task_index[pos] = idx

    # Build expected per-episode mapping from audit dict + task map
    expected_episode_idx = {}
    block_expected = []
    for key, pos in audit.items():
        m = re.fullmatch(r"ep_(\d+)_(\d+)", key)
        if not m:
            continue
        start, end = int(m.group(1)), int(m.group(2))
        if pos not in position_to_task_index:
            raise RuntimeError(f"Position {pos} from audit not found in tasks prompts")
        target_idx = int(position_to_task_index[pos])
        block_expected.append({"key": key, "start": start, "end": end, "position": pos, "target_task_index": target_idx})
        for ep in range(start, end + 1):
            expected_episode_idx[ep] = target_idx

    # Verify meta/episodes
    ep_df = pd.read_parquet(WORKDIR / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    ep_current_idx = normalize_tasks_col(ep_df["tasks"])
    ep_mismatches = []
    for _, row in ep_df.iterrows():
        ep = int(row["episode_index"])
        if ep not in expected_episode_idx:
            continue
        cur = int(ep_current_idx.loc[_])
        exp = int(expected_episode_idx[ep])
        if cur != exp:
            ep_mismatches.append({"episode_index": ep, "current": cur, "expected": exp})

    # Verify data frame-level labels
    data_mismatch_rows = 0
    data_files_checked = 0
    data_file_details = []
    for fp in sorted((WORKDIR / "data").glob("chunk-*/*.parquet")):
        data_files_checked += 1
        df = pd.read_parquet(fp)
        mask = df["episode_index"].isin(list(expected_episode_idx.keys()))
        mismatches = 0
        if mask.any():
            expected = df.loc[mask, "episode_index"].map(expected_episode_idx).astype(int)
            current = df.loc[mask, "task_index"].astype(int)
            mismatches = int((expected != current).sum())
            data_mismatch_rows += mismatches
        data_file_details.append({"file": str(fp.relative_to(WORKDIR)), "mismatch_rows": mismatches})

    proof = {
        "workdir": str(WORKDIR),
        "blocks_expected": len(block_expected),
        "episodes_expected": len(expected_episode_idx),
        "prompt_format_issues": prompt_format_issues,
        "episode_meta_mismatches": ep_mismatches,
        "data_mismatch_rows_total": data_mismatch_rows,
        "data_files_checked": data_files_checked,
        "data_file_details": data_file_details,
        "status": "PASS" if (len(prompt_format_issues) == 0 and len(ep_mismatches) == 0 and data_mismatch_rows == 0) else "FAIL",
    }

    OUT_JSON.write_text(json.dumps(proof, indent=2), encoding="utf-8")

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(f"status: {proof['status']}\n")
        f.write(f"blocks_expected: {proof['blocks_expected']}\n")
        f.write(f"episodes_expected: {proof['episodes_expected']}\n")
        f.write(f"prompt_format_issues: {len(prompt_format_issues)}\n")
        f.write(f"episode_meta_mismatches: {len(ep_mismatches)}\n")
        f.write(f"data_mismatch_rows_total: {data_mismatch_rows}\n")
        f.write(f"data_files_checked: {data_files_checked}\n")

    print(f"status={proof['status']}")
    print(f"proof_json={OUT_JSON}")
    print(f"proof_txt={OUT_TXT}")


if __name__ == "__main__":
    main()
