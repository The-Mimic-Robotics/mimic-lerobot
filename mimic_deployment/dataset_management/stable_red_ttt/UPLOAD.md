# Upload (CLI only)

Use these exact commands after running download + sniff + patch scripts.

```bash
cd /home/a/ac_pate/mimic-lerobot

# 1) Create target dataset repo (idempotent)
hf repo create Mimic-Robotics/full_ttt_redx_stable_fixed_labels --type dataset -y

# 2) Upload local fixed copy
hf upload-large-folder Mimic-Robotics/full_ttt_redx_stable_fixed_labels \
  /speed-scratch/ac_pate/hf_work/full_ttt_redx_stable_fixed_labels \
  --repo-type dataset
```

Visualizer URL:

```text
https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=Mimic-Robotics%2Ffull_ttt_redx_stable_fixed_labels
```
