#!/usr/bin/env python3
"""
Debug script to trace T5-large adapter shape transformations
"""
import torch
import torch.nn as nn

def debug_t5_large_shapes():
    """Debug T5-large configuration that shows the 8x576 vs 96x1024 issue"""
    
    # Configuration from t5_large_nofreeze.yaml
    batch_size = 8
    n_tf = 6
    seq_len = 128
    n_features = 6
    patch_len = 16  # T5-large config uses 16, not 32
    d_model = 1024  # T5-large d_model
    
    print("=== T5-Large Adapter Shape Debugging ===")
    print(f"Input configuration:")
    print(f"  batch_size: {batch_size}")
    print(f"  n_tf: {n_tf}")
    print(f"  seq_len: {seq_len}")
    print(f"  n_features: {n_features}")
    print(f"  patch_len: {patch_len}")
    print(f"  d_model: {d_model}")
    print()
    
    # Step 1: Input tensor
    x = torch.randn(batch_size, n_tf, seq_len, n_features)
    print(f"Step 1 - Input tensor: {x.shape}")
    
    # Step 2: Calculate patches
    n_patches = seq_len // patch_len
    effective_len = n_patches * patch_len
    print(f"Step 2 - Patch calculation:")
    print(f"  n_patches: {n_patches}")
    print(f"  effective_len: {effective_len}")
    
    # Step 3: Adjust sequence length
    x = x[:, :, :effective_len, :]
    print(f"Step 3 - After length adjustment: {x.shape}")
    
    # Step 4: Reshape to patches
    x = x.view(batch_size, n_tf, n_patches, patch_len, n_features)
    print(f"Step 4 - Reshape to patches: {x.shape}")
    
    # Step 5: Flatten patches
    patch_dim = patch_len * n_features
    x = x.view(batch_size, n_tf, n_patches, patch_dim)
    print(f"Step 5 - Flatten patches: {x.shape}")
    print(f"  patch_dim: {patch_dim}")
    
    # Check if patch_dim matches the 576 in the error
    print(f"\nChecking error dimensions:")
    print(f"  patch_dim = {patch_dim} (expected 576 from error)")
    print(f"  d_model = {d_model} (expected 1024 from error)")
    print(f"  batch_size * n_patches = {batch_size * n_patches} (expected 96 from error)")
    
    # Let's check what happens with different batch sizes
    print(f"\nDifferent batch size scenarios:")
    for test_batch_size in [1, 2, 4, 8, 12, 16, 24, 32]:
        total_patches = test_batch_size * n_patches
        print(f"  batch_size={test_batch_size}: total_patches={total_patches}")
        if total_patches == 96:
            print(f"    *** MATCH! batch_size={test_batch_size} gives 96 patches ***")
    
    # Now check the actual tensor manipulations
    print(f"\nActual tensor processing:")
    
    # Reset with potential problematic batch size
    test_batch_size = 12  # 12 * 8 = 96
    x = torch.randn(test_batch_size, n_tf, seq_len, n_features)
    print(f"Test input: {x.shape}")
    
    # Process to patches
    x = x.view(test_batch_size, n_tf, n_patches, patch_len, n_features)
    x = x.view(test_batch_size, n_tf, n_patches, patch_dim)
    print(f"After patching: {x.shape}")
    
    # Process each timeframe
    for tf_idx in range(n_tf):
        tf_x = x[:, tf_idx]  # [batch, n_patches, patch_dim]
        print(f"  tf_idx {tf_idx}: tf_x.shape = {tf_x.shape}")
        
        # The issue might be here - if we're trying to process all timeframes at once
        if tf_idx == 0:  # Show the potential issue
            print(f"    This gives us: [{test_batch_size}, {n_patches}, {patch_dim}]")
            print(f"    Linear layer expects: [*, {patch_dim}] -> [*, {d_model}]")
            
            # But what if we accidentally flatten the first two dimensions?
            flattened = tf_x.view(-1, patch_dim)
            print(f"    If flattened incorrectly: {flattened.shape}")
            print(f"    This would be: [{test_batch_size * n_patches}, {patch_dim}] = [{test_batch_size * n_patches}, {patch_dim}]")
    
    # Check if this matches the error pattern
    print(f"\nError pattern analysis:")
    print(f"  Expected from error: [8, 576] vs [96, 1024]")
    print(f"  Our calculation: tf_x = [{test_batch_size}, {n_patches}, {patch_dim}]")
    print(f"  If batch_size=8: tf_x = [8, {n_patches}, {patch_dim}]")
    print(f"  If flattened: [8 * {n_patches}, {patch_dim}] = [{8 * n_patches}, {patch_dim}]")
    
    # Check the 576 mystery
    print(f"\nMystery dimension 576:")
    print(f"  patch_dim = {patch_dim} (should be 96, not 576)")
    print(f"  576 / 96 = {576 / 96}")
    print(f"  576 / 6 = {576 / 6} (n_features)")
    print(f"  576 / 16 = {576 / 16} (patch_len)")
    print(f"  576 = 16 * 36 = 16 * 6 * 6")
    print(f"  Could 576 be patch_len * n_features * n_tf = {patch_len * n_features * n_tf}?")
    
    # Check if someone is accidentally using all timeframes in patch_dim
    wrong_patch_dim = patch_len * n_features * n_tf
    print(f"\nIf patch_dim incorrectly includes all timeframes:")
    print(f"  wrong_patch_dim = {patch_len} * {n_features} * {n_tf} = {wrong_patch_dim}")
    print(f"  This matches the error's 576!")
    
    # The correct fix would be to ensure we don't mix timeframes in patch_dim
    print(f"\nCorrect vs Incorrect processing:")
    print(f"  Correct: patch_dim = patch_len * n_features = {patch_len} * {n_features} = {patch_dim}")
    print(f"  Incorrect: patch_dim = patch_len * n_features * n_tf = {patch_len} * {n_features} * {n_tf} = {wrong_patch_dim}")

if __name__ == "__main__":
    debug_t5_large_shapes()