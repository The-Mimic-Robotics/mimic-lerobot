from pathlib import Path

from huggingface_hub import HfApi
from tqdm.auto import tqdm

SOURCE_DIR = Path("/speed-scratch/ac_pate/hf_work/full_ttt_redx_stable_fixed_labels")
REPO_ID = "Mimic-Robotics/full_ttt_redx_stable_fixed_labels"


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if rel.parts and rel.parts[0] == ".cache":
            continue
        yield path, rel.as_posix()


def main():
    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", private=False, exist_ok=True)

    existing = set(api.list_repo_files(repo_id=REPO_ID, repo_type="dataset"))
    all_files = list(iter_files(SOURCE_DIR))

    uploaded = 0
    skipped = 0
    failed = []

    for local_path, path_in_repo in tqdm(all_files, desc="Uploading files", unit="file"):
        if path_in_repo in existing:
            skipped += 1
            continue
        try:
            api.upload_file(
                repo_id=REPO_ID,
                repo_type="dataset",
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                commit_message=f"Upload {path_in_repo}",
            )
            uploaded += 1
        except Exception as exc:
            failed.append((path_in_repo, str(exc)))

    print(f"repo={REPO_ID}")
    print(f"total_local_files={len(all_files)}")
    print(f"uploaded={uploaded}")
    print(f"skipped_existing={skipped}")
    print(f"failed={len(failed)}")
    for item, err in failed[:20]:
        print(f"FAIL {item}: {err}")


if __name__ == "__main__":
    main()
