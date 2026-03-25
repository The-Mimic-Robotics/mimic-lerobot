import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

SCRIPT_DIR = Path("/home/a/ac_pate/mimic-lerobot/mimic_deployment/dataset_management/stable_red_ttt")
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import REQUIRED_REMOTE_FILES, SOURCE_REPO, WORKDIR

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(it, **kwargs):
        return it


def main():
    api = HfApi()
    files = api.list_repo_files(repo_id=SOURCE_REPO, repo_type="dataset")
    file_set = set(files)

    missing = [path for path in REQUIRED_REMOTE_FILES if path not in file_set]
    if missing:
        raise RuntimeError(f"Missing required files on {SOURCE_REPO}: {missing}")

    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
    WORKDIR.mkdir(parents=True, exist_ok=True)

    for path in tqdm(files, desc="Downloading dataset files", unit="file"):
        hf_hub_download(
            repo_id=SOURCE_REPO,
            repo_type="dataset",
            filename=path,
            local_dir=str(WORKDIR),
            local_dir_use_symlinks=False,
        )

    print(f"downloaded_repo={SOURCE_REPO}")
    print(f"downloaded_files={len(files)}")
    print(f"local_dir={WORKDIR}")


if __name__ == "__main__":
    main()
