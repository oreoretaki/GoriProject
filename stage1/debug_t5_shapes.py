#!/usr/bin/env python3
"""
Debug script to trace T5 adapter shape transformations
"""
import torch
import torch.nn as nn

# Simulate the PatchEmbedding forward pass
def debug_patch_embedding_shapes():
    """Debug the exact shape transformations in PatchEmbedding"""
    
    # Configuration from t5_frozen_all_10ep.yaml
    batch_size = 8
    n_tf = 6
    seq_len = 128
    n_features = 6
    patch_len = 32
    d_model = 512  # T5-base d_model
    
    print("=== T5 Adapter Shape Debugging ===")
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
    
    # Step 6: Apply linear projections per timeframe
    print(f"Step 6 - Linear projections:")
    patch_projections = nn.ModuleList([
        nn.Linear(patch_dim, d_model) for _ in range(n_tf)
    ])
    
    patches = []
    for tf_idx in range(n_tf):
        tf_x = x[:, tf_idx]  # [batch, n_patches, patch_dim]
        print(f"  tf_idx {tf_idx}: tf_x.shape = {tf_x.shape}")
        
        # Check what the linear layer expects vs what it gets
        linear_layer = patch_projections[tf_idx]
        print(f"  Linear layer {tf_idx}: in_features={linear_layer.in_features}, out_features={linear_layer.out_features}")
        
        try:
            tf_patches = linear_layer(tf_x)  # [batch, n_patches, d_model]
            print(f"  tf_patches.shape = {tf_patches.shape}")
            patches.append(tf_patches)
        except Exception as e:
            print(f"  ERROR in tf_idx {tf_idx}: {e}")
            print(f"  Expected input shape: [*, {patch_dim}]")
            print(f"  Got input shape: {tf_x.shape}")
            return
    
    patches = torch.stack(patches, dim=1)  # [batch, n_tf, n_patches, d_model]
    print(f"Step 7 - Final patches: {patches.shape}")
    
    # Check the specific issue: 8x576 vs 96x1024
    print(f"\n=== Analyzing the 8x576 vs 96x1024 issue ===")
    
    # For T5 processing, we need to reshape patches for T5 encoder
    # T5 expects: [batch, sequence_length, d_model]
    
    # If we process each timeframe separately:
    total_patches_per_tf = n_patches
    total_patches_all_tf = n_tf * n_patches
    
    print(f"Patches per timeframe: {total_patches_per_tf}")
    print(f"Total patches (all timeframes): {total_patches_all_tf}")
    
    # The issue might be in how we pass data to T5
    print(f"\nFor T5 processing:")
    print(f"  Each timeframe separately: [{batch_size}, {n_patches}, {d_model}]")
    print(f"  All timeframes together: [{batch_size}, {total_patches_all_tf}, {d_model}]")
    
    # Let's check if the issue is in the matrix multiplication
    print(f"\nMatrix multiplication check:")
    print(f"  tf_x: {tf_x.shape} = [{batch_size}, {n_patches}, {patch_dim}]")
    print(f"  Linear weight: [{d_model}, {patch_dim}]")
    print(f"  Expected output: [{batch_size}, {n_patches}, {d_model}]")
    
    # Potential issue: if patch_dim is wrong
    expected_patch_dim = patch_len * n_features
    print(f"\nPatch dimension verification:")
    print(f"  patch_len: {patch_len}")
    print(f"  n_features: {n_features}")
    print(f"  expected_patch_dim: {expected_patch_dim}")
    print(f"  actual_patch_dim: {patch_dim}")
    
    # The numbers 8x576 and 96x1024 suggest:
    # 8x576: batch_size=8, something=576
    # 96x1024: batch_size*n_patches=96, d_model=1024
    print(f"\nAnalyzing the specific error numbers:")
    print(f"  8x576: batch_size={batch_size}, mystery_dim=576")
    print(f"  96x1024: batch_size*n_patches={batch_size*n_patches}, d_model_large=1024")
    print(f"  576 = 24*24 = 18*32 = 16*36 = 12*48 = 8*72 = 6*96 = 4*144 = 3*192 = 2*288 = 1*576")
    print(f"  Could 576 be patch_dim with different config?")

if __name__ == "__main__":
    debug_patch_embedding_shapes()