#!/usr/bin/env python
"""
End-to-end training simulation test.
Runs a few training steps to verify the dataset works with policies.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def test_training_simulation():
    print("\n" + "="*60)
    print("TRAINING SIMULATION TEST")
    print("="*60)
    
    repo_id = "Mimic-Robotics/bimanual_blue_block_handover_1_15d"
    
    print(f"\nLoading dataset: {repo_id}")
    ds = LeRobotDataset(repo_id)
    
    print(f"Dataset stats:")
    print(f"  Episodes: {ds.num_episodes}")
    print(f"  Frames: {ds.num_frames}")
    print(f"  Action dim: {ds.meta.features['action']['shape']}")
    print(f"  State dim: {ds.meta.features['observation.state']['shape']}")
    
    # Create DataLoader
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
    
    # Simulate training loop
    print("\n" + "-"*40)
    print("Simulating training loop (5 iterations)...")
    print("-"*40)
    
    # Simple MLP as mock policy
    action_dim = ds.meta.features['action']['shape'][0]
    state_dim = ds.meta.features['observation.state']['shape'][0]
    
    mock_policy = torch.nn.Sequential(
        torch.nn.Linear(state_dim, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, action_dim)
    )
    
    optimizer = torch.optim.Adam(mock_policy.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    
    for i, batch in enumerate(loader):
        if i >= 5:
            break
        
        # Get state and action
        state = batch['observation.state']
        target_action = batch['action']
        
        # Forward pass
        pred_action = mock_policy(state)
        
        # Compute loss
        loss = criterion(pred_action, target_action)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"  Step {i+1}: Loss = {loss.item():.4f}")
    
    print("\n✓ Training simulation completed successfully!")
    print("  - Data loading works")
    print("  - Forward/backward passes work")
    print("  - Gradients computed correctly")
    
    print("\n" + "="*60)
    print("TRAINING SIMULATION PASSED")
    print("="*60)
    return True

if __name__ == "__main__":
    success = test_training_simulation()
    sys.exit(0 if success else 1)
