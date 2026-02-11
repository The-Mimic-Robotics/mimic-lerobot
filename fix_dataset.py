import json
from huggingface_hub import snapshot_download, HfApi

# --- CONFIG ---
SOURCE_REPO = "Mimic-Robotics/mimic_mobile_bimanual_drift_v2"
MY_REPO = "moveit2/mimic_drift_fix"
# ----------------

api = HfApi()

print(f"1. Downloading {SOURCE_REPO}...")
# Since you already downloaded it, this will be instant (it uses the cache)
local_dir = snapshot_download(repo_id=SOURCE_REPO, repo_type="dataset")

print(f"1.5 Creating repository {MY_REPO}...")
# THIS IS THE NEW LINE THAT FIXES THE 404 ERROR
api.create_repo(repo_id=MY_REPO, repo_type="dataset", exist_ok=True)

print(f"2. Uploading to {MY_REPO}...")
api.upload_folder(
    folder_path=local_dir,
    repo_id=MY_REPO,
    repo_type="dataset",
    commit_message="Cloning dataset to fix missing tags"
)

print("3. Checking version info...")
try:
    with open(f"{local_dir}/meta/info.json", "r") as f:
        info = json.load(f)
        version = info.get("codebase_version", "v2.0")
        print(f"   Found codebase_version: {version}")
except FileNotFoundError:
    print("   Warning: info.json not found! Defaulting to v2.0")
    version = "v2.0"

print(f"4. Creating tag '{version}' on your repo...")
api.create_tag(MY_REPO, tag=version, repo_type="dataset")

print(f"✅ DONE! You can now train using: {MY_REPO}")