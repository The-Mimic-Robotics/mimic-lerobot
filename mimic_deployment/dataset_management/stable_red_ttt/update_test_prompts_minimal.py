from pathlib import Path
import json
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
import importlib.util

repo = "Mimic-Robotics/full_ttt_redx_stable"
base = Path("/home/a/ac_pate/mimic-lerobot/mimic_deployment/dataset_management/stable_red_ttt")
spec = importlib.util.spec_from_file_location("instruction_normalizer", base / "instruction_normalizer.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

api = HfApi()

tasks_local = hf_hub_download(repo_id=repo, repo_type="dataset", filename="meta/tasks.parquet")
df = pd.read_parquet(tasks_local).reset_index()
name_col = "task" if "task" in df.columns else ("__index_level_0__" if "__index_level_0__" in df.columns else df.columns[0])
idx_col = "task_index" if "task_index" in df.columns else ("index" if "index" in df.columns else df.columns[1])

pairs = sorted([(int(r[idx_col]), mod.canonical_instruction_from_legacy_task(str(r[name_col]))) for _, r in df.iterrows()])
new_df = pd.DataFrame({"task_index": [i for i, _ in pairs]}, index=[t for _, t in pairs])
new_df.index.name = "__index_level_0__"
new_tasks = base / "audit" / "_new_tasks.parquet"
new_df.to_parquet(new_tasks)
api.upload_file(path_or_fileobj=str(new_tasks), path_in_repo="meta/tasks.parquet", repo_id=repo, repo_type="dataset", commit_message="Canonical uppercase cell-token prompts")

info_local = hf_hub_download(repo_id=repo, repo_type="dataset", filename="meta/info.json")
info = json.load(open(info_local, "r", encoding="utf-8"))
if isinstance(info.get("tasks"), list):
    info["tasks"] = [t for _, t in pairs]
    new_info = base / "audit" / "_new_info.json"
    json.dump(info, open(new_info, "w", encoding="utf-8"), indent=2)
    api.upload_file(path_or_fileobj=str(new_info), path_in_repo="meta/info.json", repo_id=repo, repo_type="dataset", commit_message="Sync canonical prompts in info.json")

print("Updated prompts in", repo)
for i, t in pairs:
    print(i, t)
