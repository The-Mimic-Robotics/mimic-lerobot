import json
import os
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

from config import BASE, WORKDIR

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(it, **kwargs):
        return it


def normalize_tasks_col(series: pd.Series) -> pd.Series:
    def to_int(value):
        v = value
        if hasattr(v, "tolist"):
            v = v.tolist()
        if isinstance(v, (list, tuple)) and len(v) > 0:
            v = v[0]
        return int(v)

    return series.map(to_int)


def main():
    tasks_path = WORKDIR / "meta" / "tasks.parquet"
    episodes_path = WORKDIR / "meta" / "episodes" / "chunk-000" / "file-000.parquet"

    tasks_df = pd.read_parquet(tasks_path)
    tf = tasks_df.reset_index()
    task_name_col = "task" if "task" in tf.columns else ("__index_level_0__" if "__index_level_0__" in tf.columns else tf.columns[0])
    task_idx_col = "task_index" if "task_index" in tf.columns else ("index" if "index" in tf.columns else tf.columns[1])
    task_map = {int(row[task_idx_col]): str(row[task_name_col]) for _, row in tf.iterrows()}

    eps = pd.read_parquet(episodes_path).sort_values("episode_index").reset_index(drop=True)
    eps["task_index_norm"] = normalize_tasks_col(eps["tasks"])

    blocks = []
    start = 0
    for i in range(1, len(eps) + 1):
        at_end = i == len(eps)
        gap = (not at_end) and (int(eps.loc[i, "episode_index"]) != int(eps.loc[i - 1, "episode_index"]) + 1)
        changed = (not at_end) and (int(eps.loc[i, "task_index_norm"]) != int(eps.loc[i - 1, "task_index_norm"]))
        if at_end or gap or changed:
            start_ep = int(eps.loc[start, "episode_index"])
            end_ep = int(eps.loc[i - 1, "episode_index"])
            idx = int(eps.loc[start, "task_index_norm"])
            blocks.append({
                "start": start_ep,
                "end": end_ep,
                "task_index": idx,
                "task": task_map.get(idx, "<missing>"),
                "len": end_ep - start_ep + 1,
            })
            start = i

    data_files = sorted((WORKDIR / "data").glob("chunk-*/*.parquet"))
    data_summaries = []
    for fp in tqdm(data_files, desc="Sniffing data files", unit="file"):
        df = pd.read_parquet(fp)
        data_summaries.append(
            {
                "file": str(fp.relative_to(WORKDIR)),
                "rows": int(len(df)),
                "episode_min": int(df["episode_index"].min()),
                "episode_max": int(df["episode_index"].max()),
                "task_index_unique": sorted([int(x) for x in df["task_index"].dropna().unique().tolist()]),
            }
        )

    out = {
        "workdir": str(WORKDIR),
        "task_map": task_map,
        "blocks": blocks,
        "data_files": data_summaries,
    }
    output_json = BASE / "audit" / "05_sniff_report.json"
    output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"blocks_found={len(blocks)}")
    print(f"data_files={len(data_files)}")
    print(f"report={output_json}")


if __name__ == "__main__":
    main()
