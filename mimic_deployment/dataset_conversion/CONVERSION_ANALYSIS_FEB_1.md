# Dataset Conversion Analysis: Bimanual → Mobile Bimanual (New Format)


## Files:

OLD FORMAT (v2.1, 12D):
  jetson_bimanual_recording_test: cameras=[wrist_right, wrist_left, realsense_top]
  bimanual_blue_block_handover_20: cameras=[wrist_right, wrist_left, realsense_top]
  bimanual_blue_block_handover_21: cameras=[wrist_right, wrist_left, realsense_top]
  bimanual_blue_block_handover_22: cameras=[wrist_right, wrist_left, realsense_top]
  bimanual_blue_block_handover_23: cameras=[wrist_right, wrist_left, realsense_top]
  bimanual_blue_block_handover_24: cameras=[wrist_right, wrist_left, realsense_top]
  bimanual_blue_block_handover_25: cameras=[wrist_right, wrist_left, realsense_top]
  bimanual_blue_block_handover_26: cameras=[wrist_right, wrist_left, realsense_top]
  bimanual_blue_block_handover_1: cameras=[wrist_right, wrist_left, realsense_top]
  bimanual_blue_block_handover_2: cameras=[wrist_right, wrist_left, realsense_top]
  bimanual_blue_block_handover_3: cameras=[wrist_right, wrist_left, realsense_top]
  bimanual_blue_block_handover_4: cameras=[wrist_right, wrist_left, realsense_top]
  bimanual_blue_block_handover_5: cameras=[wrist_right, wrist_left, realsense_top]
  bimanual_blue_block_handover_6: cameras=[wrist_right, wrist_left, realsense_top]
  bimanual_blue_block_handover_7: cameras=[wrist_right, wrist_left, realsense_top]
  bimanual_blue_block_handover_16: cameras=[wrist_right, wrist_left, realsense_top]
  bimanual_blue_block_handover_18: cameras=[wrist_right, wrist_left, realsense_top]
  bimanual_blue_block_handover_19: cameras=[wrist_right, wrist_left, realsense_top]
  bimanual_blue_block_handover_14: cameras=[wrist_right, wrist_left, realsense_top]
  bimanual_blue_block_handover_15: cameras=[wrist_right, wrist_left, realsense_top]
  bimanual_blue_block_handover_17: cameras=[wrist_right, wrist_left, realsense_top]

NEW FORMAT (v3.0, 15D):
  mimic_mobile_bimanual_drift_v2: cameras=[right_wrist, left_wrist, front, top]
  mobile_bimanual_blue_block_handover_1: cameras=[wrist_right, wrist_left, realsense_top]
  mobile_bimanual_blue_block_handover_2: cameras=[wrist_right, wrist_left, realsense_top]
  mobile_bimanual_blue_block_handover_3: cameras=[wrist_right, wrist_left, realsense_top]
  mobile_bimanual_blue_block_handover_4: cameras=[wrist_right, wrist_left, realsense_top]
  mobile_bimanual_blue_block_handover_5: cameras=[wrist_right, wrist_left, realsense_top]
  mobile_bimanual_blue_block_handover_6: cameras=[wrist_right, wrist_left, realsense_top]
  mobile_bimanual_blue_block_handover_7: cameras=[wrist_right, wrist_left, realsense_top]
  mobile_bimanual_blue_block_handover_14: cameras=[wrist_right, wrist_left, realsense_top]
  mobile_bimanual_blue_block_handover_15: cameras=[wrist_right, wrist_left, realsense_top]
  mobile_bimanual_blue_block_handover_16: cameras=[wrist_right, wrist_left, realsense_top]
  mobile_bimanual_blue_block_handover_17: cameras=[wrist_right, wrist_left, realsense_top]
  mobile_bimanual_blue_block_handover_18: cameras=[wrist_right, wrist_left, realsense_top]
  mobile_bimanual_blue_block_handover_19: cameras=[wrist_right, wrist_left, realsense_top]
  mobile_bimanual_blue_block_handover_20: cameras=[wrist_right, wrist_left, realsense_top]
  mobile_bimanual_blue_block_handover_21: cameras=[wrist_right, wrist_left, realsense_top]
  mobile_bimanual_blue_block_handover_22: cameras=[wrist_right, wrist_left, realsense_top]
  mobile_bimanual_blue_block_handover_23: cameras=[wrist_right, wrist_left, realsense_top]
  mobile_bimanual_blue_block_handover_24: cameras=[wrist_right, wrist_left, realsense_top]
  mobile_bimanual_blue_block_handover_25: cameras=[wrist_right, wrist_left, realsense_top]
  mobile_bimanual_blue_block_handover_26: cameras=[wrist_right, wrist_left, realsense_top]
  mimic_displacement_to_handover_blue_block_v2: cameras=[right_wrist, left_wrist, front, top]
  mimic_displacement_to_handover_blue_block_v7: cameras=[right_wrist, left_wrist, front, top]
  mimic_displacement_to_handover_blue_block_v8: cameras=[right_wrist, left_wrist, front, top]
  mimic_displacement_to_handover_blue_block_v6: cameras=[right_wrist, left_wrist, front, top]
  mimic_displacement_to_handover_blue_block_with_new_hands_v2: cameras=[right_wrist, left_wrist, front, top]
  mimic_displacement_to_handover_blue_block_with_new_hands_v3: cameras=[right_wrist, left_wrist, front, top]

Datasets with TOP camera:
  mimic_mobile_bimanual_drift_v2
  mimic_displacement_to_handover_blue_block_v2
  mimic_displacement_to_handover_blue_block_v7
  mimic_displacement_to_handover_blue_block_v8
  mimic_displacement_to_handover_blue_block_v6
  mimic_displacement_to_handover_blue_block_with_new_hands_v2
  mimic_displacement_to_handover_blue_block_with_new_hands_v3

## Executive Summary

Converting `bimanual_blue_block_handover_*` (OLD) to match `mimic_displacement_*` (NEW) format requires:
1. **Version upgrade**: v2.1 → v3.0
2. **Action space expansion**: 12D → 15D (add 3 base velocities, all zeros)
3. **Observation space expansion**: 12D → 15D (add 3 base poses, all zeros)
4. **Camera renaming + addition**: 3 cameras → 4 cameras (rename + add dummy)
5. **Directory restructure**: v2.1 to v3.0 file organization
6. **Robot type update**: `bi_so101_follower` → `mimic_follower`

**Status of existing mobile_bimanual datasets**: FAILED/BROKEN
- Camera names don't match
- Camera indices wrong
- Cannot be used for training

## 1. Detailed Differences

### 1.1 Dataset Versions
```
OLD: codebase_version = "v2.1"
NEW: codebase_version = "v3.0"
```

### 1.2 Robot Type
```
OLD: robot_type = "bi_so101_follower"  (stationary bimanual)
NEW: robot_type = "mimic_follower"     (mobile bimanual)
```

### 1.3 Action Space (Motors)
```
OLD: 12 dimensions
  - left_shoulder_pan.pos
  - left_shoulder_lift.pos
  - left_elbow_flex.pos
  - left_wrist_flex.pos
  - left_wrist_roll.pos
  - left_gripper.pos
  - right_shoulder_pan.pos
  - right_shoulder_lift.pos
  - right_elbow_flex.pos
  - right_wrist_flex.pos
  - right_wrist_roll.pos
  - right_gripper.pos

NEW: 15 dimensions (same 12 + 3 new)
  - [same 12 as above]
  - base_vx        ← ADD (zero-padded)
  - base_vy        ← ADD (zero-padded)
  - base_omega     ← ADD (zero-padded)
```

### 1.4 Observation State
```
OLD: 12 dimensions (same joint names as action)

NEW: 15 dimensions
  - [same 12 joint positions]
  - base_x         ← ADD (zero-padded)
  - base_y         ← ADD (zero-padded)
  - base_theta     ← ADD (zero-padded)
```

### 1.5 Camera Configuration

**CRITICAL CHANGES:**

#### OLD (3 cameras):
```
1. observation.images.wrist_right    (640x480)
2. observation.images.wrist_left     (640x480)
3. observation.images.realsense_top  (640x480) - RealSense D435
```

#### NEW (4 cameras):
```
1. observation.images.right_wrist    (640x480) - renamed from wrist_right
2. observation.images.left_wrist     (640x480) - renamed from wrist_left
3. observation.images.front          (640x480) - NEW webcam ← DUMMY NEEDED
4. observation.images.top            (1280x720) - ZED camera (replaced RealSense)
```

**Camera Mapping Strategy:**
```
OLD wrist_right    → NEW right_wrist  (rename)
OLD wrist_left     → NEW left_wrist   (rename)
OLD realsense_top  → NEW top          (rename, keep video)
[NO SOURCE]        → NEW front        (create dummy/blank video)
```

**Camera Resolution Change:**
- `top` camera: 640x480 (RealSense) → 1280x720 (ZED)
- Need to decide: upscale old footage OR keep at 640x480 OR add black bars

### 1.6 File Path Structure

#### OLD (v2.1):
```
data/chunk-000/episode_000000.parquet
videos/chunk-000/wrist_right/episode_000000.mp4
videos/chunk-000/wrist_left/episode_000000.mp4
videos/chunk-000/realsense_top/episode_000000.mp4
```

#### NEW (v3.0):
```
data/chunk-000/file-000.parquet
videos/right_wrist/chunk-000/file-000.mp4
videos/left_wrist/chunk-000/file-000.mp4
videos/front/chunk-000/file-000.mp4
videos/top/chunk-000/file-000.mp4
```

**Note:** v3.0 reorganized directory structure significantly.

## 2. What Needs to Be Done

### Step 1: Version Conversion (v2.1 → v3.0)
**Use existing utility:** `src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py`

This handles:
- Directory restructuring
- Metadata format updates
- Episodes stats generation
- Data/video path reorganization

### Step 2: Action/Observation Space Expansion
**Edit parquet files:**
- Read each `.parquet` data file
- Extract `action` column (shape: N x 12)
- Zero-pad to shape: N x 15
- Extract `observation.state` column (shape: N x 12)  
- Zero-pad to shape: N x 15
- Write back to parquet

**Update metadata:**
- `info.json`: Update action/obs.state shapes and names
- Add base velocity/position names to feature lists

### Step 3: Camera Renaming + Video Manipulation
**A. Rename existing cameras:**
```python
wrist_right    → right_wrist
wrist_left     → left_wrist
realsense_top  → top
```

**B. Handle top camera resolution mismatch:**
Three options:
1. **Keep original 640x480** (easiest, but inconsistent with new datasets)
2. **Upscale to 1280x720** (consistent, but introduces artificial resolution)
3. **Add black bars to 1280x720** (consistent, preserves original, recommended)

**C. Create dummy `front` camera videos:**
Must create blank/black videos with:
- Resolution: 640x480
- FPS: 30
- Duration: Match corresponding episode duration
- Codec: av1, yuv420p (to match existing videos)
- One video per episode

### Step 4: Update Metadata Files
**info.json:**
- `codebase_version`: "v2.1" → "v3.0"
- `robot_type`: "bi_so101_follower" → "mimic_follower"
- `features.action.shape`: [12] → [15]
- `features.action.names`: Add base velocities
- `features.observation.state.shape`: [12] → [15]
- `features.observation.state.names`: Add base positions
- Rename camera keys:
  - `observation.images.wrist_right` → `observation.images.right_wrist`
  - `observation.images.wrist_left` → `observation.images.left_wrist`
  - `observation.images.realsense_top` → `observation.images.top`
- Add new camera:
  - `observation.images.front` with 640x480 specs
- Update `observation.images.top` resolution: 640x480 → 1280x720 (if upscaling)

**episodes.jsonl, tasks.jsonl:**
- No changes needed (episode metadata is version-agnostic)

**episodes_stats.jsonl:**
- Regenerate after conversion (use LeRobot utilities)

## 3. Technical Challenges

### Challenge 1: Creating Dummy Videos
**Problem:** No built-in utility to create blank videos matching episode durations.

**Solution:** Use ffmpeg or Python video libraries (opencv-python, imageio-ffmpeg):
```python
# Pseudo-code
for each episode:
    duration = get_episode_duration(episode_id)
    create_black_video(
        output_path=f"videos/front/chunk-{chunk}/file-{idx}.mp4",
        width=640,
        height=480,
        fps=30,
        duration=duration,
        codec="libsvtav1"  # av1 codec
    )
```

### Challenge 2: Top Camera Resolution Handling
**Problem:** Old datasets have 640x480 RealSense, new datasets have 1280x720 ZED.

**Recommended approach:** Add black padding bars to 1280x720
```python
# Pseudo-code
for each top_camera_video:
    video = load_video(old_path)
    padded_video = add_letterboxing(
        video, 
        target_width=1280,
        target_height=720
    )
    save_video(padded_video, new_path)
```

**Alternative:** Keep 640x480 (simpler but creates inconsistency in training).

### Challenge 3: Large Dataset Size
**Problem:** 20+ datasets × ~25 episodes each × 3 video streams = ~1500 videos to process.

**Solution:** 
- Process in batches
- Use multiprocessing for video operations
- Estimate storage: Each dataset ~500MB-2GB, total ~20-40GB

### Challenge 4: Parquet Column Type Consistency
**Problem:** Zero-padding changes array dimensions, must maintain PyArrow schema.

**Solution:**
```python
# Use pyarrow to ensure correct typing
new_action = pa.array([
    np.pad(action, (0, 3), mode='constant') 
    for action in old_actions
], type=pa.list_(pa.float32(), 15))
```

## 4. Existing Conversion Scripts Analysis

### Script 1: `mimic_deployment/dataset_conversion/convert_bimanual_to_mobile.py`
**What it does:**
- ✅ Zero-pads actions (12D → 15D)
- ✅ Zero-pads obs.state (12D → 15D)
- ✅ Updates info.json metadata
- ✅ Copies video files
- ❌ Does NOT rename cameras
- ❌ Does NOT create dummy front camera
- ❌ Does NOT handle resolution mismatch
- ❌ Does NOT do v2.1 → v3.0 conversion first

**Status:** INCOMPLETE - This is why mobile_bimanual datasets are broken!

### Script 2: `src/mimic/scripts/convert_old_to_mobile_dataset.py`
**What it does:**
- ✅ Calls v2.1 → v3.0 converter
- ✅ Zero-pads actions/observations
- ❌ Does NOT handle cameras
- ❌ Does NOT create dummy videos

**Status:** INCOMPLETE - Also doesn't handle camera issues.

### Script 3: `src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py` (LeRobot official)
**What it does:**
- ✅ Version conversion (v2.1 → v3.0)
- ✅ Directory restructuring
- ✅ Stats regeneration
- ❌ Does NOT modify action dimensions
- ❌ Does NOT handle camera changes

**Status:** PERFECT for step 1, but need custom logic for steps 2-3.

## 5. Recommended Implementation Strategy

### Phase 1: Build Custom Conversion Pipeline
Create new script: `convert_bimanual_to_mobile_complete.py`

**Architecture:**
```
1. Version Convert (use existing LeRobot utility)
   ↓
2. Action/Obs Expansion (custom code with pyarrow)
   ↓
3. Camera Rename + Video Copy (custom code)
   ↓
4. Dummy Video Generation (ffmpeg wrapper)
   ↓
5. Resolution Handling (optional: opencv/ffmpeg)
   ↓
6. Metadata Update (json manipulation)
   ↓
7. Stats Regeneration (LeRobot utility)
   ↓
8. Upload to Hub (huggingface_hub API)
```

### Phase 2: Test on Single Dataset
**Test dataset:** `Mimic-Robotics/bimanual_blue_block_handover_1`
**Output:** `Mimic-Robotics/achal_mobile_bimanual_1`

**Validation checklist:**
- [ ] Can load with LeRobotDataset
- [ ] Action shape is (15,)
- [ ] Observation.state shape is (15,)
- [ ] 4 cameras present: right_wrist, left_wrist, front, top
- [ ] All videos play correctly
- [ ] Front camera is black/dummy
- [ ] Top camera resolution is 1280x720 (or documented as 640x480)
- [ ] Training runs without errors

### Phase 3: Batch Convert All Datasets
Once validated, process all bimanual datasets:
```
bimanual_blue_block_handover_1  → achal_mobile_bimanual_1
bimanual_blue_block_handover_2  → achal_mobile_bimanual_2
...
bimanual_blue_block_handover_26 → achal_mobile_bimanual_26
```

**Automation:** Create batch script to loop through all datasets.

## 6. Estimated Work Breakdown

### Development Time:
1. **Dummy video creation utility:** 2-4 hours
   - Write function to create blank av1 videos
   - Match LeRobot video encoding specs
   - Test with various durations

2. **Camera renaming + resolution handling:** 2-3 hours
   - File path manipulation
   - Optional: letterboxing/padding code
   - Metadata updates

3. **Integration into conversion pipeline:** 3-5 hours
   - Combine all steps
   - Error handling
   - Progress tracking
   - Validation checks

4. **Testing on single dataset:** 2-3 hours
   - Run conversion
   - Validate output
   - Test training
   - Fix issues

5. **Batch processing setup:** 1-2 hours
   - Parallel processing
   - Resource management
   - Upload automation

**Total:** ~12-20 hours of development + 5-10 hours testing/debugging

### Computational Time:
- **Single dataset conversion:** ~10-30 minutes
  - Depends on: episode count, video sizes, CPU cores
- **All 20 datasets:** ~4-8 hours with parallel processing

### Storage Requirements:
- **Working space:** ~50-100GB (temp processing)
- **Final datasets:** ~20-40GB (depending on compression)

## 7. Gotchas & Edge Cases

1. **Episode gaps:** Some datasets have missing episodes (1-7, then 14-26)
   - Handle gracefully, don't assume sequential

2. **Video sync:** Ensure dummy front videos exactly match frame counts
   - Use episode duration, not approximate

3. **Codec compatibility:** av1 encoding can be slow
   - Consider parallel encoding
   - Or use h264 as faster alternative (but larger files)

4. **Metadata consistency:** All 4 scripts must agree on:
   - Camera names
   - Feature dimensions
   - Feature names order

5. **Hub upload limits:** HuggingFace has file size limits
   - May need to use git-lfs
   - Check quotas

6. **Training compatibility:** After conversion, test that:
   - ACT policy loads correctly
   - Multi-dataset training works
   - No shape mismatches

## 8. Alternative Approaches

### Option A: Minimal Conversion (Fastest)
- Keep 640x480 for top camera
- Skip dummy front camera creation entirely
- Only expand action/obs to 15D
- **Trade-off:** Incompatible with new datasets for multi-dataset training

### Option B: Full Conversion (Recommended)
- Full v2.1 → v3.0
- Camera renaming
- Dummy front camera
- Letterbox top to 1280x720
- **Trade-off:** More work, but fully compatible

### Option C: Hybrid Approach
- Convert action/obs/version
- Rename cameras
- Add dummy front camera
- Keep top at 640x480 but update metadata to document this
- **Trade-off:** Balanced, mostly compatible

## 9. Validation & Testing Plan

### Unit Tests:
- [ ] Dummy video creation (correct duration, fps, codec)
- [ ] Action padding (12D → 15D with correct zeros)
- [ ] Observation padding (12D → 15D with correct zeros)
- [ ] Camera renaming (file paths + metadata)
- [ ] Metadata updates (all fields correct)

### Integration Tests:
- [ ] Full pipeline on test dataset
- [ ] Load with LeRobotDataset
- [ ] Video playback works
- [ ] Shape assertions pass

### Training Tests:
- [ ] Single converted dataset trains
- [ ] Multi-dataset with old+new trains
- [ ] No shape mismatches
- [ ] Evaluation runs

## 10. Next Steps & Recommendations

### Immediate Actions:
1. **Decide on top camera strategy:** Keep 640x480 vs letterbox to 1280x720?
2. **Choose naming convention:** Stick with `achal_mobile_bimanual_*`?
3. **Set up development environment:** Install ffmpeg, test video libraries
4. **Create dummy video utility** (can be standalone script first)

### Before Writing Code:
1. Download 1-2 test datasets locally
2. Manually inspect parquet structure
3. Test dummy video creation with one episode
4. Verify you can load/modify parquet with pyarrow

### Code Development Order:
1. Write dummy video generator (standalone)
2. Write camera renaming logic (standalone)
3. Combine with existing conversion scripts
4. Add validation checks
5. Test on single dataset end-to-end
6. Create batch processing wrapper

### Post-Conversion:
1. Update `TRAINING_COMMANDS.txt` with new dataset names
2. Update `datasets.yaml` with new groups
3. Test multi-dataset training with mixed old+new
4. Document conversion process for team

## 11. Questions to Answer Before Starting

1. **Top camera resolution:** Keep 640x480 or letterbox to 1280x720?
2. **Front camera content:** Pure black or add watermark/text indicating dummy?
3. **Naming convention:** `achal_mobile_bimanual_X` confirmed?
4. **Upload location:** Push to Mimic-Robotics org or personal first?
5. **Version tagging:** Tag as v3.0 immediately or test first?
6. **Batch size:** Convert all 20 at once or start with subset?
7. **Existing mobile_bimanual datasets:** Delete or keep for reference?
8. **Storage budget:** Confirm HuggingFace quota can handle 20-40GB

---

**SUMMARY:** This is a well-defined but non-trivial conversion task. The main complexity is video manipulation (dummy videos + optional resolution handling). The action/observation padding is straightforward. The existing scripts provide good foundation but are incomplete. Budget 12-20 hours dev time + 4-8 hours conversion time.
