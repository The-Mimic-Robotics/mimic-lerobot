#!/usr/bin/env python3

import argparse
import re
import shutil
from pathlib import Path

import wandb
from huggingface_hub import HfApi

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def sanitize_repo_name(name: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
    return clean.strip("_")


def extract_step_from_artifact_name(name: str) -> str | None:
    name_no_version = name.split(":", 1)[0]
    matches = re.findall(r"(\d{4,6})", name_no_version)
    if not matches:
        return None
    return matches[-1].zfill(6)


def copy_tree_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def download_artifact_with_progress(artifact, target_root: Path) -> Path:
    target_root.mkdir(parents=True, exist_ok=True)
    artifact_files = list(artifact.files())

    if not artifact_files:
        return Path(artifact.download(root=str(target_root)))

    if tqdm is not None:
        iterator = tqdm(artifact_files, total=len(artifact_files), desc=f"download {artifact.name}", unit="file")
    else:
        print(f"[info] downloading {artifact.name} ({len(artifact_files)} files)")
        iterator = artifact_files

    for artifact_file in iterator:
        artifact_file.download(root=str(target_root), replace=True)

    return target_root


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recover W&B model artifacts and upload them to Hugging Face as checkpoints and final model."
    )
    parser.add_argument(
        "--wandb-run-path",
        required=True,
        help="W&B run path in the form entity/project/run_id",
    )
    parser.add_argument(
        "--hf-repo-id",
        default="",
        help="Target HF model repo id (default: Mimic-Robotics/<wandb_run_name_sanitized>)",
    )
    parser.add_argument(
        "--workspace",
        default="/tmp/wandb_to_hf_recovery",
        help="Temporary local workspace",
    )
    parser.add_argument(
        "--repo-owner",
        default="Mimic-Robotics",
        help="Default owner used when --hf-repo-id is omitted",
    )
    parser.add_argument(
        "--no-upload-final",
        action="store_true",
        help="Only upload recovered checkpoints; skip final-model root upload",
    )
    parser.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="Upload all recovered checkpoints instead of only the latest",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print actions without uploading",
    )
    parser.add_argument(
        "--clean-workspace",
        action="store_true",
        help="Delete workspace before starting recovery",
    )

    args = parser.parse_args()

    api = wandb.Api()
    run = api.run(args.wandb_run_path)

    if args.hf_repo_id:
        hf_repo_id = args.hf_repo_id
    else:
        hf_repo_id = f"{args.repo_owner}/{sanitize_repo_name(run.name)}"

    workspace = Path(args.workspace).expanduser().resolve()
    if args.clean_workspace and workspace.exists():
        shutil.rmtree(workspace)
    artifacts_root = workspace / "downloaded_artifacts"
    recovered_root = workspace / "recovered"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    recovered_root.mkdir(parents=True, exist_ok=True)

    model_artifacts = [a for a in run.logged_artifacts() if a.type == "model"]
    if not model_artifacts:
        print(f"[error] No model artifacts found in run: {args.wandb_run_path}")
        return 2

    recoverable_artifacts: list[tuple[int, str, object]] = []

    print(f"[info] W&B run: {args.wandb_run_path}")
    print(f"[info] HF repo target: {hf_repo_id}")
    print(f"[info] Model artifacts found: {len(model_artifacts)}")

    for artifact in model_artifacts:
        step_str = extract_step_from_artifact_name(artifact.name)
        if step_str is None:
            print(f"[warn] Could not infer step from artifact name, skipping: {artifact.name}")
            continue
        recoverable_artifacts.append((int(step_str), step_str, artifact))

    if not recoverable_artifacts:
        print("[error] No recoverable model artifacts were found.")
        return 3

    recoverable_artifacts.sort(key=lambda x: x[0])
    selected_artifacts = recoverable_artifacts if args.all_checkpoints else [recoverable_artifacts[-1]]

    print(
        "[info] Artifacts selected for download: "
        + ", ".join([f"{step_str}:{artifact.name}" for _, step_str, artifact in selected_artifacts])
    )

    recovered_steps: list[tuple[int, Path]] = []

    for step_int, step_str, artifact in selected_artifacts:
        local_artifact_dir = download_artifact_with_progress(
            artifact, artifacts_root / artifact.name.replace(":", "_")
        )
        ckpt_dir = recovered_root / "checkpoints" / step_str / "pretrained_model"
        copy_tree_contents(local_artifact_dir, ckpt_dir)
        recovered_steps.append((step_int, ckpt_dir))
        print(f"[info] Recovered artifact {artifact.name} -> checkpoints/{step_str}/pretrained_model")

    recovered_steps.sort(key=lambda x: x[0])
    latest_step, latest_ckpt_dir = recovered_steps[-1]

    hf = HfApi()
    if not args.dry_run:
        hf.create_repo(repo_id=hf_repo_id, repo_type="model", exist_ok=True)

    upload_candidates = recovered_steps

    for step_int, ckpt_dir in upload_candidates:
        step_name = str(step_int).zfill(6)
        path_in_repo = f"checkpoints/{step_name}/pretrained_model"
        print(f"[upload] {ckpt_dir} -> {hf_repo_id}/{path_in_repo}")
        if not args.dry_run:
            hf.upload_folder(
                repo_id=hf_repo_id,
                repo_type="model",
                folder_path=str(ckpt_dir),
                path_in_repo=path_in_repo,
                commit_message=f"Recover checkpoint {step_name} from W&B artifact",
            )

    if not args.no_upload_final:
        print(f"[upload-final] latest checkpoint step={latest_step:06d} -> {hf_repo_id}/")
        if not args.dry_run:
            hf.upload_folder(
                repo_id=hf_repo_id,
                repo_type="model",
                folder_path=str(latest_ckpt_dir),
                path_in_repo=".",
                commit_message=f"Recover final model from W&B artifact step {latest_step:06d}",
            )

    print("[done] Recovery upload completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
