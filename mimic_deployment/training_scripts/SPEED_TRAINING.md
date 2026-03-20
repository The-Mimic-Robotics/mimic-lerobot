# SPEED Cluster Training (Simple, SBATCH-only)

This guide is for running training from `speed-submit` using the dedicated script:

- `mimic_deployment/training_scripts/train_manager_speed.sh`

No `srun` mode. No mixed local/SPEED logic. One script dedicated to SPEED.

## 1) Where to run this from

Run training commands from `speed-submit` (not from your `salloc` VS Code server shell).

```bash
ssh ac_pate@speed.encs.concordia.ca
cd /home/a/ac_pate/mimic-lerobot
```

## 2) Activate conda first (required)

Run this before any auth checks:

```bash
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
source /speed-scratch/$USER/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate lerobot
```

If activation fails, your conda path is different; locate `conda.sh` and source it first.

## 3) Auth checks (works even if CLI wrappers are missing)

```bash
python -m wandb --version
python -m wandb login

python -m huggingface_hub.commands.huggingface_cli login

test -f ~/.cache/huggingface/token && echo "HF token found"
test -f ~/.netrc && echo "netrc found"
```

If `wandb` command is missing, this is normal on some setups; use `python -m wandb ...`.

Optional environment variables:

```bash
export WANDB_PROJECT="mimic-lerobot"
export WANDB_ENTITY="mathias1-misc-robotics"
```

## 4) Make script executable

```bash
chmod +x mimic_deployment/training_scripts/train_manager_speed.sh
```

## 5) Simple command (hassle-free)

```bash
./mimic_deployment/training_scripts/train_manager_speed.sh \
  --policy xvla \
  --dataset-group redx_full_vlm
```

What you get:
- submits with `sbatch`
- streams the job log in your terminal
- writes a plain `.log` file in `outputs/logs/`
- prints job id, dataset group, log path, and W&B URL (when it appears)

## 5) Advanced command (more control)

```bash
./mimic_deployment/training_scripts/train_manager_speed.sh \
  --policy xvla,pi05 \
  --dataset-group redx_full_vlm,most_recent \
  --steps 300000 \
  --batch-size 32 \
  --gpus 2 \
  --slurm-mem 256G
```

## 6) Useful options

- `--list-groups`
- `--list-policies`
- `--dry-run`
- `--no-follow` (submit jobs but do not stream logs in terminal)

## 7) Logs and monitoring

```bash
squeue -u "$USER"
ls -lth outputs/logs/*.log | head
```

If needed, manually follow a job log:

```bash
tail -f outputs/logs/<job_name>.log
```

## 8) Common failures

1. **W&B run appears then cancels**
   - ensure token/login is valid on SPEED (`python -m wandb login`)
   - ensure job log has no auth/network failures

2. **HF access errors**
   - ensure `python -m huggingface_hub.commands.huggingface_cli login` is done on SPEED

3. **`wandb: Command not found` or `huggingface-cli: Command not found`**
   - activate conda env first (`conda activate lerobot`)
   - use `python -m ...` forms shown above

4. **OOM**
   - lower `--batch-size` or `--gpus`

5. **No logs in terminal**
   - remove `--no-follow`
   - check job is in queue and log file exists under `outputs/logs/`
