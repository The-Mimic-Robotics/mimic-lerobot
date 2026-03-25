import json
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

repo = "Mimic-Robotics/full_ttt_redx_stable"
out = "/home/a/ac_pate/mimic-lerobot/mimic_deployment/dataset_management/stable_red_ttt/audit/00_structure_report.txt"
json_out = "/home/a/ac_pate/mimic-lerobot/mimic_deployment/dataset_management/stable_red_ttt/audit/00_blocks.json"

files = list_repo_files(repo_id=repo, repo_type="dataset")

tasks_path = hf_hub_download(repo_id=repo, repo_type="dataset", filename="meta/tasks.parquet")
eps_path = hf_hub_download(repo_id=repo, repo_type="dataset", filename="meta/episodes/chunk-000/file-000.parquet")

tasks_df = pd.read_parquet(tasks_path)
eps_df = pd.read_parquet(eps_path)

tf = tasks_df.reset_index()
task_name_col = "task" if "task" in tf.columns else ("__index_level_0__" if "__index_level_0__" in tf.columns else tf.columns[0])
task_idx_col = "task_index" if "task_index" in tf.columns else ("index" if "index" in tf.columns else tf.columns[1])
task_map = {int(r[task_idx_col]): str(r[task_name_col]) for _, r in tf.iterrows()}

e = eps_df.copy()
ep_col = "episode_index" if "episode_index" in e.columns else e.columns[0]
et_col = "task_index" if "task_index" in e.columns else ("tasks" if "tasks" in e.columns else None)
if et_col is None:
    raise RuntimeError("No task column in episodes parquet")

if et_col == "tasks":
    def norm(value):
        v = value
        if hasattr(v, "tolist"):
            v = v.tolist()
        if isinstance(v, (list, tuple)) and len(v) > 0:
            v = v[0]
        return int(v)
    e["task_index_norm"] = e[et_col].map(norm)
else:
    e["task_index_norm"] = e[et_col].astype(int)

e = e.sort_values(ep_col).reset_index(drop=True)
blocks = []
start_i = 0
for i in range(1, len(e) + 1):
    at_end = i == len(e)
    episode_gap = (not at_end) and (int(e.loc[i, ep_col]) != int(e.loc[i - 1, ep_col]) + 1)
    task_changed = (not at_end) and (int(e.loc[i, "task_index_norm"]) != int(e.loc[i - 1, "task_index_norm"]))
    if at_end or episode_gap or task_changed:
        start_ep = int(e.loc[start_i, ep_col])
        end_ep = int(e.loc[i - 1, ep_col])
        task_idx = int(e.loc[start_i, "task_index_norm"])
        blocks.append(
            {
                "start": start_ep,
                "end": end_ep,
                "task_index": task_idx,
                "task": task_map.get(task_idx, "<missing>"),
                "len": end_ep - start_ep + 1,
            }
        )
        start_i = i

data_files = sorted([f for f in files if f.startswith("data/") and f.endswith(".parquet")])
sample_idxs = sorted(set([0, len(data_files) // 2, len(data_files) - 1] if data_files else []))
samples = []
for idx in sample_idxs:
    fp = data_files[idx]
    lp = hf_hub_download(repo_id=repo, repo_type="dataset", filename=fp)
    df = pd.read_parquet(lp)
    cols = list(df.columns)
    samples.append(
        {
            "file": fp,
            "rows": int(len(df)),
            "cols": cols[:12] + (["..."] if len(cols) > 12 else []),
            "episode_min": int(df["episode_index"].min()) if "episode_index" in cols else None,
            "episode_max": int(df["episode_index"].max()) if "episode_index" in cols else None,
            "task_index_unique": sorted([int(x) for x in df["task_index"].dropna().unique().tolist()])[:30]
            if "task_index" in cols else None,
        }
    )

with open(out, "w", encoding="utf-8") as f:
    f.write(f"Dataset: {repo}\n\n")
    f.write("[1] meta/tasks.parquet\n")
    f.write(f"- local_path: {tasks_path}\n")
    f.write(f"- columns: {list(tasks_df.columns)}\n")
    f.write(f"- index_name: {tasks_df.index.name}\n")
    f.write(f"- num_rows(tasks): {len(tasks_df)}\n")
    f.write("- task_index -> task string mapping:\n")
    for key in sorted(task_map):
        f.write(f"  {key}: {task_map[key]}\n")

    f.write("\n[2] meta/episodes/chunk-000/file-000.parquet\n")
    f.write(f"- local_path: {eps_path}\n")
    f.write(f"- columns: {list(eps_df.columns)}\n")
    f.write(f"- num_rows(episodes): {len(eps_df)}\n")
    f.write(f"- episode_index col: {ep_col}\n")
    f.write(f"- task column detected: {et_col} (normalized to task_index_norm)\n")
    f.write("- chain: meta/episodes.task_index_norm and data/*.parquet.task_index both resolve via meta/tasks.parquet.task_index -> task string\n")

    f.write("\n[3] sample data parquet files\n")
    for sample in samples:
        f.write(
            f"- {sample['file']}: rows={sample['rows']} episode_range={sample['episode_min']}..{sample['episode_max']}\n"
        )
        f.write(f"  cols(head): {sample['cols']}\n")
        f.write(f"  task_index_unique(sample): {sample['task_index_unique']}\n")

    f.write("\n[4] task_index block boundaries in episodes metadata\n")
    f.write(f"- blocks_found: {len(blocks)}\n")
    for block in blocks:
        f.write(
            f"  ep_{block['start']}_{block['end']} | len={block['len']} | task_index={block['task_index']} | label={block['task']}\n"
        )

with open(json_out, "w", encoding="utf-8") as f:
    json.dump({"repo": repo, "task_map": task_map, "blocks": blocks}, f, indent=2)

print(f"Wrote {out}")
print(f"Wrote {json_out}")
print(f"blocks={len(blocks)}")
