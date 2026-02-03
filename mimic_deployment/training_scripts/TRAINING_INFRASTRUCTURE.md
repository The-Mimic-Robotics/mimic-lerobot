# Mimic Robot Multi-Policy Training Infrastructure

Complete training infrastructure for training multiple VLA models (xVLA, Pi0.5, Pi0, Groot, ACT, Diffusion) on Mimic robot datasets across multiple computers.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Dataset Groups](#dataset-groups)
- [Policy Types](#policy-types)
- [Computer Configurations](#computer-configurations)
- [Training Scripts](#training-scripts)
- [Usage Examples](#usage-examples)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Train a model in 3 steps:

```bash
# 1. List available dataset groups
./mimic_deployment/train_manager.sh --list-groups

# 2. List available policies
./mimic_deployment/train_manager.sh --list-policies

# 3. Start training
./mimic_deployment/train_manager.sh --policy xvla --dataset-group all_datasets
```

---

## 📊 Dataset Groups

All datasets are in **v3.0 format** (15D action space, 4 cameras: `right_wrist`, `left_wrist`, `front`, `top`).

### Available Groups

| Group Name | Datasets | Description |
|------------|----------|-------------|
| `all_datasets` | 27 | All available datasets (6 displacement + 20 drift + 1 drift_v2) |
| `displacement_only` | 7 | Mobile base displacement-to-handover tasks |
| `drift_only` | 20 | Drift-corrected converted datasets (stationary) |
| `high_quality` | 16 | Best quality datasets with new hands and drift correction |
| `stationary` | 20 | Stationary bimanual handover (drift-corrected) |
| `most_recent` | 10 | Latest recordings with best quality |
| `new_hands` | 2 | Datasets with improved gripper control |
| `old_hands` | 25 | Datasets recorded with original hands |

### Dataset Details

View the complete dataset configuration:
```bash
cat mimic_deployment/dataset_groups.yaml
```

List datasets in a specific group:
```bash
python3 mimic_deployment/dataset_groups.py all_datasets --format list
```

---

## 🤖 Policy Types

### 1. **xVLA** - X-VLA Vision-Language-Action Model
- **Priority:** ⭐⭐⭐⭐⭐ (Highest)
- **Base Model:** `lerobot/xvla-base`
- **Features:** Multi-camera, language-conditioned, soft-prompted
- **Batch Size:** 8 (Odin), 4 (Jupiter/Mathias)
- **Training Time:** ~20,000 steps

### 2. **Pi0.5** - Physical Intelligence π₀.₅
- **Priority:** ⭐⭐⭐⭐⭐ (Highest)
- **Base Model:** `lerobot/pi05_base`
- **Features:** Flow matching, open-world generalization, auto normalization
- **Batch Size:** 2 (Odin), 1 (Jupiter/Mathias)
- **Training Time:** ~10,000 steps
- **Note:** Automatically handles quantile normalization

### 3. **Groot** - NVIDIA GR00T N1.5
- **Priority:** ⭐⭐⭐⭐ (High)
- **Base Model:** Pre-trained GR00T checkpoint
- **Features:** Humanoid foundation model, multi-GPU support
- **Batch Size:** 4 (Odin), 2 (Jupiter/Mathias)
- **Training Time:** ~15,000 steps
- **Special:** Auto-detects GPU count via `nvidia-smi`

### 4. **Pi0** - Physical Intelligence π₀
- **Priority:** ⭐⭐⭐ (Medium)
- **Base Model:** `lerobot/pi0_base`
- **Features:** Flow matching, cross-embodiment training
- **Batch Size:** 2 (Odin), 1 (Jupiter/Mathias)
- **Training Time:** ~10,000 steps

### 5. **ACT** - Action Chunking with Transformers
- **Priority:** ⭐⭐ (Lower)
- **Features:** Lightweight, fast training, transformer-based
- **Batch Size:** 8 (Odin), 6 (Jupiter/Mathias)
- **Training Time:** ~100,000 steps

### 6. **Diffusion** - Diffusion Policy
- **Priority:** ⭐⭐ (Lower)
- **Features:** Diffusion-based action generation
- **Batch Size:** 8 (Odin), 6 (Jupiter/Mathias)
- **Training Time:** ~100,000 steps

---

## 💻 Computer Configurations

### Odin (Primary - Pi Models)
- **GPU:** RTX 3090 Ti (22GB VRAM)
- **Hostname:** `odin` or `ODIN-IEEE`
- **Use Case:** All policies, primary for Pi models
- **Default Batch Sizes:**
  - xVLA: 8
  - Pi0/Pi0.5: 2
  - Groot: 4
  - ACT/Diffusion: 8

### Jupiter
- **GPU:** RTX 5070 (12GB VRAM)
- **Hostname:** `jupiter`
- **Use Case:** All policies
- **Default Batch Sizes:**
  - xVLA: 4
  - Pi0/Pi0.5: 1
  - Groot: 2
  - ACT/Diffusion: 6

### Mathias
- **GPU:** RTX 3080 Ti (10GB VRAM)
- **Hostname:** `mathias`
- **Use Case:** All policies, primarily ACT and Diffusion
- **Default Batch Sizes:**
  - xVLA: 4
  - Pi0/Pi0.5: 1
  - Groot: 2
  - ACT/Diffusion: 6

---

## 📝 Training Scripts

### Directory Structure

```
mimic_deployment/
├── dataset_groups.yaml          # Dataset group definitions
├── dataset_groups.py            # Dataset resolver utility
├── train_manager.sh             # Master orchestrator
└── training_scripts/
    ├── xvla/
    │   ├── train.sh             # xVLA training
    │   └── eval.sh              # xVLA evaluation
    ├── pi05/
    │   ├── train.sh             # Pi0.5 training
    │   └── eval.sh              # Pi0.5 evaluation
    ├── pi0/
    │   ├── train.sh             # Pi0 training
    │   └── eval.sh              # Pi0 evaluation
    ├── groot/
    │   ├── train.sh             # Groot training (multi-GPU)
    │   └── eval.sh              # Groot evaluation
    ├── act/
    │   ├── train.sh             # ACT training
    │   └── eval.sh              # ACT evaluation
    └── diffusion/
        ├── train.sh             # Diffusion training
        └── eval.sh              # Diffusion evaluation
```

### Smart Features

All training scripts include:
- ✅ Auto computer detection via `$COMPUTER` or hostname
- ✅ Dataset group resolution
- ✅ Auto-generated job names and output directories
- ✅ Auto-generated Hugging Face repo IDs
- ✅ Background execution with nohup
- ✅ Log file management in `outputs/logs/`
- ✅ WandB integration with auto-notes
- ✅ Computer-specific batch size defaults
- ✅ PID tracking for process management

---

## 🎯 Usage Examples

### Using the Master Orchestrator (Recommended)

```bash
# Train xVLA on all datasets
./mimic_deployment/train_manager.sh --policy xvla --dataset-group all_datasets

# Train Pi0.5 on high quality datasets
./mimic_deployment/train_manager.sh --policy pi05 --dataset-group high_quality

# Train Groot on most recent datasets
./mimic_deployment/train_manager.sh --policy groot --dataset-group most_recent

# Train ACT on stationary datasets with custom batch size
./mimic_deployment/train_manager.sh --policy act --dataset-group stationary --batch-size 4

# Dry run to see configuration
./mimic_deployment/train_manager.sh --policy xvla --dataset-group all_datasets --dry-run
```

### Direct Script Execution

```bash
# Train xVLA directly
export DATASET_GROUP=all_datasets
./mimic_deployment/training_scripts/xvla/train.sh

# Train Pi0.5 with custom settings
DATASET_GROUP=high_quality BATCH_SIZE=4 ./mimic_deployment/training_scripts/pi05/train.sh

# Train on specific computer
COMPUTER=jupiter DATASET_GROUP=most_recent ./mimic_deployment/training_scripts/groot/train.sh
```

### Evaluation

```bash
# Set evaluation parameters
export MODEL_PATH=Mimic-Robotics/xvla_odin_all_datasets
export EVAL_DATASET=Mimic-Robotics/eval_xvla_test
export NUM_EPISODES=10
export TASK="Bimanual blue block handover from left to right hand"

# Run evaluation
./mimic_deployment/training_scripts/xvla/eval.sh
```

---

## 🔧 Advanced Usage

### Custom Training Parameters

All scripts support environment variable overrides:

```bash
# Custom steps and save frequency
STEPS=50000 SAVE_FREQ=10000 ./mimic_deployment/train_manager.sh --policy xvla --dataset-group all_datasets

# Custom batch size and workers
BATCH_SIZE=16 NUM_WORKERS=20 ./mimic_deployment/train_manager.sh --policy act --dataset-group stationary

# Multiple overrides
COMPUTER=jupiter \
DATASET_GROUP=high_quality \
BATCH_SIZE=2 \
NUM_WORKERS=8 \
STEPS=30000 \
./mimic_deployment/training_scripts/pi05/train.sh
```

### Monitoring Training

```bash
# Find your training log
ls -lth outputs/logs/

# Monitor training in real-time
tail -f outputs/logs/xvla_odin_all_datasets.log

# Check training status
ps aux | grep lerobot-train

# Check GPU usage
watch -n 1 nvidia-smi
```

### Managing Training Jobs

```bash
# Kill specific training
kill <PID>

# Kill all training jobs
pkill -f lerobot-train

# Check training outputs
ls -lth outputs/train/

# View checkpoints
ls -lth outputs/train/xvla_odin_all_datasets/checkpoints/
```

### Creating Custom Dataset Groups

Edit `mimic_deployment/dataset_groups.yaml`:

```yaml
groups:
  my_custom_group:
    - Mimic-Robotics/dataset1
    - Mimic-Robotics/dataset2
    - neryotw/dataset3
```

Then use:
```bash
./mimic_deployment/train_manager.sh --policy xvla --dataset-group my_custom_group
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Dataset group not found
```bash
# List available groups
./mimic_deployment/train_manager.sh --list-groups

# Check dataset resolver
python3 mimic_deployment/dataset_groups.py --list-groups
```

#### 2. Training script not found
```bash
# Ensure scripts are executable
chmod +x mimic_deployment/training_scripts/*/train.sh
chmod +x mimic_deployment/training_scripts/*/eval.sh
```

#### 3. Out of memory errors
```bash
# Reduce batch size
BATCH_SIZE=1 ./mimic_deployment/train_manager.sh --policy pi05 --dataset-group all_datasets

# Reduce number of workers
NUM_WORKERS=4 ./mimic_deployment/train_manager.sh --policy xvla --dataset-group all_datasets
```

#### 4. Groot multi-GPU issues
```bash
# Check GPU detection
nvidia-smi

# Force single GPU
NUM_GPUS=1 ./mimic_deployment/training_scripts/groot/train.sh
```

#### 5. Camera configuration issues
All training scripts automatically handle 4-camera setup with correct names:
- `right_wrist`
- `left_wrist`
- `front`
- `top`

If datasets have different camera names, the rename_map handles it automatically.

### Log File Locations

- **Training Logs:** `outputs/logs/<policy>_<computer>_<dataset_group>.log`
- **Training Outputs:** `outputs/train/<policy>_<computer>_<dataset_group>/`
- **Checkpoints:** `outputs/train/<policy>_<computer>_<dataset_group>/checkpoints/`

### Getting Help

```bash
# Show train_manager help
./mimic_deployment/train_manager.sh --help

# List all options for a training script
head -50 mimic_deployment/training_scripts/xvla/train.sh
```

---

## 📦 Dependencies

### Required Packages
- `lerobot` (main package)
- `accelerate` (for Groot multi-GPU training)
- `pyyaml` (for config parsing)
- `wandb` (for experiment tracking)

### Installation

```bash
# Main environment
pip install -e .

# Accelerate (for Groot)
pip install accelerate

# Optional: Pi0/Pi0.5 dependencies
pip install -e ".[pi]"

# Optional: Groot dependencies  
pip install -e ".[groot]"
```

---

## 🎓 Best Practices

### Training Priority Order
1. **xVLA** and **Pi0.5** (highest priority, best generalization)
2. **Groot** (strong humanoid capabilities)
3. **Pi0** (solid baseline)
4. **ACT** and **Diffusion** (lightweight alternatives)

### Dataset Selection
- Start with `high_quality` or `most_recent` for best results
- Use `all_datasets` for maximum diversity
- Use `stationary` or `displacement_only` for specific behaviors

### Batch Size Tuning
- Start with defaults
- Increase until OOM, then reduce by 1-2
- Monitor GPU memory with `nvidia-smi`

### Multi-Computer Workflow
```bash
# On Odin (RTX 3090 Ti) - Train Pi models
./mimic_deployment/train_manager.sh --policy pi05 --dataset-group all_datasets

# On Jupiter (RTX 5070) - Train xVLA
./mimic_deployment/train_manager.sh --policy xvla --dataset-group high_quality

# On Mathias (RTX 3080 Ti) - Train ACT
./mimic_deployment/train_manager.sh --policy act --dataset-group stationary
```

---

## 📄 License

This training infrastructure is part of the Mimic Robot project.

## 🤝 Contributing

When adding new policies:
1. Create directory: `mimic_deployment/training_scripts/<policy>/`
2. Add `train.sh` and `eval.sh` scripts
3. Follow the existing smart script pattern
4. Update this README
5. Test with `--dry-run` flag

---

**Happy Training! 🚀**
