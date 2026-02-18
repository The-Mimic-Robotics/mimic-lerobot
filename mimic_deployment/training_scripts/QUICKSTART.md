# Quick Start Guide - Mimic Robot Training (Multi-Group + Auto-Upload)

## 🎯 Training in 60 Seconds
hf upload Mimic-Robotics/act_odin_red_x_handover_20k outputs/train/act_odin_red_x_handover_and_place_center/checkpoints/020000
hf upload Mimic-Robotics/mimic_tictactoe_red_x_handover_v3 ~/.cache/huggingface/lerobot/Mimic-Robotics/mimic_tictactoe_red_x_handover_v3 --repo-type dataset
### Step 1: Choose Your Setup
```bash
# List available dataset groups
./mimic_deployment/training_scripts/train_manager.sh --list-groups

# List available policies
./mimic_deployment/training_scripts/train_manager.sh --list-policies
```

### Step 2: Train Your Model
```bash
# Train xVLA on all datasets (recommended for first run)
./mimic_deployment/training_scripts/train_manager.sh --policy xvla --dataset-group all_datasets

# Train Pi0.5 on high quality datasets (recommended for Pi models)
./mimic_deployment/training_scripts/train_manager.sh --policy pi05 --dataset-group high_quality

# Train ACT on stationary datasets
./mimic_deployment/training_scripts/train_manager.sh --policy act --dataset-group stationary

# 🆕 NEW: Train on multiple dataset groups sequentially
./mimic_deployment/training_scripts/train_manager.sh --policy xvla --dataset-group high_quality,most_recent --wait-for-completion

# 🆕 NEW: Train multiple policies on multiple groups with auto-upload
./mimic_deployment/training_scripts/train_manager.sh --policy xvla,pi05 --dataset-group high_quality,most_recent --push-to-hub --wait-for-completion
```

### Step 3: Monitor Progress
```bash
# Find your log file
ls -lth outputs/logs/

# Watch training in real-time
tail -f outputs/logs/xvla_odin_all_datasets.log
```

---

## 📊 Quick Reference

### Recommended Combinations

| Priority | Policy | Dataset Group | Computer | Why? |
|----------|--------|---------------|----------|------|
| 1️⃣ | `xvla` | `all_datasets` | Odin | Best generalization, most data |
| 2️⃣ | `pi05` | `high_quality` | Odin | Latest model, best quality data |
| 3️⃣ | `groot` | `most_recent` | Odin | Humanoid foundation, recent data |
| 4️⃣ | `pi0` | `all_datasets` | Odin | Solid baseline |
| 5️⃣ | `act` | `stationary` | Mathias | Fast, lightweight |
| 6️⃣ | `wall_oss` | `high_quality` | Odin | 🆕 Embodied foundation model |

### Computer Assignments

- **Odin (RTX 3090 Ti, 22GB):** Train Pi models (pi05, pi0), xVLA, Wall-OSS
- **Jupiter (RTX 5070, 12GB):** Train xVLA, ACT, Wall-OSS
- **Mathias (RTX 3080 Ti, 10GB):** Train ACT, Wall-OSS

---

## 🔥 Common Training Commands

### Train on Different Dataset Groups
```bash
# All available datasets (27 datasets)
./mimic_deployment/training_scripts/train_manager.sh --policy xvla --dataset-group all_datasets

# High quality datasets (16 datasets)
./mimic_deployment/training_scripts/train_manager.sh --policy pi05 --dataset-group high_quality

# Most recent datasets (10 datasets)
./mimic_deployment/training_scripts/train_manager.sh --policy groot --dataset-group most_recent

# Stationary bimanual only (20 datasets)
./mimic_deployment/training_scripts/train_manager.sh --policy act --dataset-group stationary

# Mobile displacement tasks (7 datasets)
./mimic_deployment/training_scripts/train_manager.sh --policy xvla --dataset-group displacement_only

# 🆕 NEW: Train on multiple groups sequentially
./mimic_deployment/training_scripts/train_manager.sh --policy xvla --dataset-group high_quality,most_recent,new_hands
```

### 🆕 Multi-Group Sequential Training
```bash
# Train xVLA on high_quality, then most_recent, then all_datasets
./mimic_deployment/training_scripts/train_manager.sh --policy xvla --dataset-group high_quality,most_recent,all_datasets --wait-for-completion

# Train multiple policies on multiple groups (comprehensive experiment)
./mimic_deployment/training_scripts/train_manager.sh --policy xvla,pi05,groot --dataset-group high_quality,most_recent --push-to-hub --wait-for-completion
```

### Custom Batch Sizes
```bash
# Reduce batch size if OOM
./mimic_deployment/training_scripts/train_manager.sh --policy pi05 --dataset-group all_datasets --batch-size 1

# Increase batch size for faster training
./mimic_deployment/training_scripts/train_manager.sh --policy act --dataset-group stationary --batch-size 12
```

### Custom Training Steps
```bash
# Quick test run (1000 steps)
./mimic_deployment/training_scripts/train_manager.sh --policy xvla --dataset-group high_quality --steps 1000

# Extended training (50000 steps)
./mimic_deployment/training_scripts/train_manager.sh --policy act --dataset-group all_datasets --steps 50000
```

---

## 🚦 Training Status

### Check What's Running
```bash
# See active training jobs
ps aux | grep lerobot-train

# See GPU usage
watch -n 1 nvidia-smi

# Check all log files
ls -lth outputs/logs/
```

### Stop Training
```bash
# Find PID from log output, then:
kill <PID>

# Or stop all training
pkill -f lerobot-train
```

---

## 📁 Output Locations

```
outputs/
├── logs/                                    # Training logs
│   ├── xvla_odin_all_datasets.log
│   ├── pi05_odin_high_quality.log
│   └── ...
└── train/                                   # Training outputs
    ├── xvla_odin_all_datasets/
    │   ├── checkpoints/                     # Model checkpoints
    │   │   ├── 005000/
    │   │   ├── 010000/
    │   │   └── ...
    │   └── train_config.json
    └── ...
```

---

## 🆘 Quick Troubleshooting

### Problem: Out of Memory (OOM)
**Solution:** Reduce batch size
```bash
./mimic_deployment/train_manager.sh --policy pi05 --dataset-group all_datasets --batch-size 1
```

### Problem: Training too slow
**Solution:** Reduce number of workers or use smaller dataset group
```bash
./mimic_deployment/train_manager.sh --policy xvla --dataset-group high_quality --num-workers 4
```

### Problem: Can't find dataset group
**Solution:** List available groups
```bash
./mimic_deployment/train_manager.sh --list-groups
```

### Problem: Wrong computer detected
**Solution:** Override computer name
```bash
./mimic_deployment/train_manager.sh --policy xvla --dataset-group all_datasets --computer jupiter
```

---

## 🎓 Training Strategy

### For Maximum Performance
1. Start with `xvla` on `high_quality` (best generalization)
2. Train `pi05` on `all_datasets` (latest tech)
3. Train `groot` on `most_recent` (foundation model)

### For Quick Results
1. Train `act` on `stationary` (fast, lightweight)
2. Use smaller dataset groups (`high_quality`, `most_recent`)
3. Reduce training steps to 10000

### For Experimentation
1. Use `--dry-run` to test configurations
2. Start with smaller groups (`new_hands`, `most_recent`)
3. Use lower batch sizes to fit more experiments on one GPU

---

## 📚 Full Documentation

For comprehensive documentation, see:
- [TRAINING_INFRASTRUCTURE.md](TRAINING_INFRASTRUCTURE.md) - Complete guide
- [dataset_groups.yaml](dataset_groups.yaml) - Dataset definitions
- Individual training scripts in `training_scripts/*/`

---

**Ready to train? Pick a command above and get started! 🚀**



#custom ACT

./mimic_deployment/training_scripts/train_manager.sh \
  --policy act,pi05,groot,pi0,wall_oss \
  --dataset-group red_x_handover_and_place_tictactoe_v8 \
  --batch-size 30 \
  --action-steps 30 \
  --chunk-size 50 



--------------------------------------------------------pi0

./mimic_deployment/training_scripts/train_manager.sh \
  --policy pi0 \
  --dataset-group red_x_handover_and_place_tictactoe \
  --batch-size 30 \
  --action-steps 50 \
  --chunk-size 50 
  --noback



  ./mimic_deployment/training_scripts/train_manager.sh \
  --policy act \
  --dataset-group red_x_handover_and_place_tictactoe \
  --batch-size 16 \
  --action-steps 50 \
  --chunk-size 50 \
  --steps 500000 


# 24Gb vram


./mimic_deployment/training_scripts/train_manager.sh \
  --policy act \
  --dataset-group red_x_handover_and_place_tictactoe \
  --batch-size 16 \
  --action-steps 50 \
  --chunk-size 50 \
  --steps 500000 



# 11Gb vram
./mimic_deployment/training_scripts/train_manager.sh \
  --policy act \
  --dataset-group redx_full_vlm \
  --batch-size 8 \
  --action-steps 50 \
  --chunk-size 50 \
  --steps 200000 


./mimic_deployment/training_scripts/train_manager.sh \
  --policy xvla \
  --dataset-group redx_full_vlm \
  --batch-size 8

./mimic_deployment/training_scripts/train_manager.sh \
  --policy pi05 \
  --dataset-group redx_full_vlm \
  --batch-size 4 \
  --steps 100000 



  ./mimic_deployment/training_scripts/train_manager.sh \
  --policy smolvla \
  --dataset-group redx_full_vlm \
  --batch-size 48 


  -> action step should be the number of action which it moves from and the 
  chunk size is teh number generate to predict future


  hf upload Mimic-Robotics/act_odin_red_x_handover_10action_20k outputs/train/act_odin_red_x_handover_and_place_tictactoe_v4/checkpoints/020000/pretrained_model



hf upload Mimic-Robotics/act_odin_red_x_30a_20k  outputs/train/act_odin_red_x_handover_and_place_tictactoe_v8/checkpoints/020000/pretrained_model
hf upload Mimic-Robotics/act_odin_red_x_30a_40k  outputs/train/act_odin_red_x_handover_and_place_tictactoe_v8/checkpoints/040000/pretrained_model
hf upload Mimic-Robotics/act_odin_red_x_30a_60k  outputs/train/act_odin_red_x_handover_and_place_tictactoe_v8/checkpoints/060000/pretrained_model
  



hf upload Mimic-Robotics/pi0_odin_redx_20k_30b_50a  outputs/train/
hf upload Mimic-Robotics/smolvla_odin_redx_100k_64b_50a  outputs/train/

hf upload Mimic-Robotics/act_math_redxVlm_50a_8b_200k  outputs/train/act_


hf upload Mimic-Robotics/act_augusto_red_x_50a_14b_100k  outputs/train/act_


hf upload Mimic-Robotics/mimic_tictactoe_red_x_handover_v3 ~/.cache/huggingface/lerobot/Mimic-Robotics/mimic_tictactoe_red_x_handover_v3 --repo-type dataset



mimic_tictactoe_blue_o_handover_center_slow_v1
mimic_tictactoe_blue_o_handover_center_slow_v2
mimic_tictactoe_blue_o_handover_top_left_slow_v1 
mimic_tictactoe_blue_o_handover_top_right_slow_v1
mimic_tictactoe_blue_o_handover_top_right_slow_v2 
mimic_tictactoe_red_x_handover_center_slow_v3 


huggingface-cli upload Mimic-Robotics/mimic_tictactoe_blue_o_handover_bottom_left_slow_v1 ~/.cache/huggingface/lerobot/Mimic-Robotics/mimic_tictactoe_blue_o_handover_bottom_left_slow_v1 --repo-type dataset


huggingface-cli upload Mimic-Robotics/mimic_tictactoe_blue_o_handover_bottom_left_slow_v2 ~/.cache/huggingface/lerobot/Mimic-Robotics/mimic_tictactoe_blue_o_handover_bottom_left_slow_v2 --repo-type dataset


huggingface-cli upload Mimic-Robotics/mimic_tictactoe_blue_o_handover_bottom_right_slow_v1 ~/.cache/huggingface/lerobot/Mimic-Robotics/mimic_tictactoe_blue_o_handover_bottom_right_slow_v1 --repo-type dataset


huggingface-cli upload Mimic-Robotics/mimic_tictactoe_blue_o_handover_bottom_right_slow_v2 ~/.cache/huggingface/lerobot/Mimic-Robotics/mimic_tictactoe_blue_o_handover_bottom_right_slow_v2 --repo-type dataset


huggingface-cli upload Mimic-Robotics/mimic_tictactoe_blue_o_handover_top_left_slow_v2 ~/.cache/huggingface/lerobot/Mimic-Robotics/mimic_tictactoe_blue_o_handover_top_left_slow_v2 --repo-type dataset

huggingface-cli upload Mimic-Robotics/mimic_tictactoe_blue_o_handover_top_left_slow_v3 ~/.cache/huggingface/lerobot/Mimic-Robotics/mimic_tictactoe_blue_o_handover_top_left_slow_v3 --repo-type dataset

huggingface-cli upload Mimic-Robotics/mimic_tictactoe_blue_o_handover_top_left_slow_v4 ~/.cache/huggingface/lerobot/Mimic-Robotics/mimic_tictactoe_blue_o_handover_top_left_slow_v4 --repo-type dataset

                       
                       
                       
                       


huggingface-cli upload Mimic-Robotics/eval_act_14b_50a_100k_v2 ~/.cache/huggingface/lerobot/Mimic-Robotics/eval_act_14b_50a_100k_v2 --repo-type model



#success
#--policy.path=outputs/train/act_MISC_red_x_handover_and_place_tictactoe_20260210_2145/checkpoints/100000/pretrained_model \


