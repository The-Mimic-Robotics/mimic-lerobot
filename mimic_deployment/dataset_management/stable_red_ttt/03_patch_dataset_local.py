import ast
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

for env_var in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "ARROW_NUM_THREADS",
]:
    os.environ.setdefault(env_var, "1")

SCRIPT_DIR = Path("/home/a/ac_pate/mimic-lerobot/mimic_deployment/dataset_management/stable_red_ttt")
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import AUDIT_MD, BASE, WORKDIR

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(it, **kwargs):
        return it


POSITION_TO_LEGACY_LABEL = {
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

LEGACY_LABEL_TO_POSITION = {value: key for key, value in POSITION_TO_LEGACY_LABEL.items()}


def parse_audit(md_text: str) -> dict[str, str]:
    match = re.search(r"audit\s*=\s*\{[\s\S]*?\}", md_text)
    if not match:
        raise RuntimeError("audit dict not found in audit guide")
    return ast.literal_eval(match.group(0).split("=", 1)[1].strip())


def normalize_tasks_col(series: pd.Series) -> pd.Series:
    def to_int(value):
        v = value
        if hasattr(v, "tolist"):
            v = v.tolist()
        if isinstance(v, (list, tuple)) and len(v) > 0:
            v = v[0]
        return int(v)

    return series.map(to_int)


def load_normalizer():
    module_path = BASE / "instruction_normalizer.py"
    spec = importlib.util.spec_from_file_location("instruction_normalizer", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def infer_position_from_task_label(label: str) -> str | None:
    clean = re.sub(r"\s+", " ", label.strip().lower())
    if clean in LEGACY_LABEL_TO_POSITION:
        return LEGACY_LABEL_TO_POSITION[clean]

    slot_match = re.search(r"<([A-Z_]+)>", label)
    if slot_match:
        slot = slot_match.group(1).strip().lower()
        if slot in POSITION_TO_LEGACY_LABEL:
            return slot
    return None


def main():
    tasks_path = WORKDIR / "meta" / "tasks.parquet"
    episodes_path = WORKDIR / "meta" / "episodes" / "chunk-000" / "file-000.parquet"

    tasks_df = pd.read_parquet(tasks_path).reset_index()
    name_col = "task" if "task" in tasks_df.columns else ("__index_level_0__" if "__index_level_0__" in tasks_df.columns else tasks_df.columns[0])
    idx_col = "task_index" if "task_index" in tasks_df.columns else ("index" if "index" in tasks_df.columns else tasks_df.columns[1])
    position_to_idx = {}
    for _, row in tasks_df.iterrows():
        label = str(row[name_col])
        idx = int(row[idx_col])
        position = infer_position_from_task_label(label)
        if position is not None:
            position_to_idx[position] = idx

    audit = parse_audit(AUDIT_MD.read_text(encoding="utf-8"))
    episode_to_idx = {}
    for key, pos in audit.items():
        m = re.fullmatch(r"ep_(\d+)_(\d+)", key)
        if not m:
            continue
        start, end = int(m.group(1)), int(m.group(2))
        target_pos = str(pos).strip()
        if target_pos not in position_to_idx:
            raise RuntimeError(f"Target position missing from tasks mapping: {target_pos}")
        target_idx = position_to_idx[target_pos]
        for ep in range(start, end + 1):
            episode_to_idx[ep] = target_idx

    frame_changes = 0
    data_files = sorted((WORKDIR / "data").glob("chunk-*/*.parquet"))
    for path in tqdm(data_files, desc="Patching data files", unit="file"):
        df = pd.read_parquet(path)
        mask = df["episode_index"].isin(list(episode_to_idx.keys()))
        if mask.any():
            expected = df.loc[mask, "episode_index"].map(episode_to_idx).astype(int)
            current = df.loc[mask, "task_index"].astype(int)
            frame_changes += int((current != expected).sum())
            df.loc[mask, "task_index"] = expected.values
            df.to_parquet(path, index=False)

    eps = pd.read_parquet(episodes_path)
    eps_current = normalize_tasks_col(eps["tasks"])
    mask = eps["episode_index"].isin(list(episode_to_idx.keys()))
    expected = eps.loc[mask, "episode_index"].map(episode_to_idx).astype(int)
    episode_changes = int((eps_current.loc[mask] != expected).sum())
    eps.loc[mask, "tasks"] = expected.map(lambda x: [int(x)])
    eps.to_parquet(episodes_path, index=False)

    normalizer = load_normalizer()
    rewritten = []
    for _, row in tasks_df.iterrows():
        old = str(row[name_col])
        position = infer_position_from_task_label(old)
        if position is None:
            raise RuntimeError(f"Could not infer position from task label: {old}")
        new = normalizer.canonical_instruction_from_position(position)
        rewritten.append((int(row[idx_col]), new))
    rewritten = sorted(rewritten)

    new_tasks = pd.DataFrame({"task_index": [i for i, _ in rewritten]}, index=[t for _, t in rewritten])
    new_tasks.index.name = "__index_level_0__"
    new_tasks.to_parquet(tasks_path)

    info_path = WORKDIR / "meta" / "info.json"
    info_updated = False
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if isinstance(info.get("tasks"), list) and len(info.get("tasks")) > 0:
            info["tasks"] = [t for _, t in rewritten]
            info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
            info_updated = True

    report = BASE / "audit" / "06_patch_report.json"
    report.write_text(
        json.dumps(
            {
                "workdir": str(WORKDIR),
                "frame_rows_changed": frame_changes,
                "episode_rows_changed": episode_changes,
                "info_updated": info_updated,
                "new_tasks": [{"task_index": i, "task": t} for i, t in rewritten],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"frame_rows_changed={frame_changes}")
    print(f"episode_rows_changed={episode_changes}")
    print(f"report={report}")


if __name__ == "__main__":
    main()
