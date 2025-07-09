#!/usr/bin/env python3
"""
Debug the exact shape transformation issue in T5 adapter
"""
import torch
import torch.nn as nn

def debug_exact_issue():
    """Debug the exact issue that produces 8x576 vs 96x1024"""
    
    # From the error analysis, we know:
    # - batch_size=12 gives 96 patches (12 * 8 = 96)
    # - patch_dim=96 (16 * 6)
    # - But error shows 576, which is 96 * 6 = 576
    
    batch_size = 12
    n_tf = 6
    seq_len = 128
    n_features = 6
    patch_len = 16
    d_model = 1024
    
    print("=== Exact Issue Analysis ===")
    print(f"Configuration: batch={batch_size}, n_tf={n_tf}, seq_len={seq_len}, n_features={n_features}")
    print(f"patch_len={patch_len}, d_model={d_model}")
    
    # Create input tensor
    x = torch.randn(batch_size, n_tf, seq_len, n_features)
    print(f"\nInput: {x.shape}")
    
    # Calculate patches
    n_patches = seq_len // patch_len
    effective_len = n_patches * patch_len
    print(f"n_patches: {n_patches}, effective_len: {effective_len}")
    
    # Process through the PatchEmbedding steps
    x = x[:, :, :effective_len, :]
    print(f"After length adjustment: {x.shape}")
    
    # Reshape to patches
    x = x.view(batch_size, n_tf, n_patches, patch_len, n_features)
    print(f"After view to patches: {x.shape}")
    
    # Flatten patches
    patch_dim = patch_len * n_features
    x = x.view(batch_size, n_tf, n_patches, patch_dim)
    print(f"After flatten patches: {x.shape}")
    print(f"patch_dim: {patch_dim}")
    
    # Now the critical step - process each timeframe
    print(f"\nProcessing each timeframe:")
    for tf_idx in range(n_tf):
        tf_x = x[:, tf_idx]  # [batch, n_patches, patch_dim]
        print(f"tf_idx {tf_idx}: tf_x.shape = {tf_x.shape}")
        
        # Check what happens if we have dimension mismatch
        if tf_idx == 0:
            print(f"  tf_x should be: [{batch_size}, {n_patches}, {patch_dim}]")
            print(f"  Linear layer expects: [*, {patch_dim}] -> [*, {d_model}]")
            
            # Create the linear layer
            linear_layer = nn.Linear(patch_dim, d_model)
            print(f"  Linear layer: in_features={linear_layer.in_features}, out_features={linear_layer.out_features}")
            
            # Test the forward pass
            try:
                output = linear_layer(tf_x)
                print(f"  SUCCESS: output.shape = {output.shape}")
            except Exception as e:
                print(f"  ERROR: {e}")
    
    # Now let's check what would cause the 8x576 vs 96x1024 error
    print(f"\n=== Investigating 8x576 vs 96x1024 error ===")
    
    # The error suggests:
    # - Input to linear layer: [8, 576]
    # - Expected by linear layer: [96, 1024]
    
    # But this doesn't make sense for a linear layer which should be:
    # - Input: [*, input_features]
    # - Weight: [output_features, input_features]
    # - Output: [*, output_features]
    
    # The issue might be in how the input is being reshaped before the linear layer
    print(f"Possible issue scenarios:")
    
    # Scenario 1: Wrong batch size being used
    wrong_batch_size = 8
    x_wrong = torch.randn(wrong_batch_size, n_tf, seq_len, n_features)
    x_wrong = x_wrong.view(wrong_batch_size, n_tf, n_patches, patch_len, n_features)
    x_wrong = x_wrong.view(wrong_batch_size, n_tf, n_patches, patch_dim)
    tf_x_wrong = x_wrong[:, 0]  # [8, 8, 96]
    print(f"Scenario 1 - Wrong batch size: tf_x_wrong.shape = {tf_x_wrong.shape}")
    
    # Scenario 2: If someone accidentally flattens all timeframes into patch_dim
    wrong_patch_dim = patch_len * n_features * n_tf
    print(f"Scenario 2 - Wrong patch_dim: {wrong_patch_dim}")
    
    # This would give us [8, 576] if we have:
    # - batch_size = 8
    # - wrong_patch_dim = 16 * 6 * 6 = 576
    
    # Scenario 3: If the linear layer is configured wrong
    wrong_linear = nn.Linear(wrong_patch_dim, d_model)
    print(f"Scenario 3 - Wrong linear layer: in_features={wrong_linear.in_features}, out_features={wrong_linear.out_features}")
    
    # The error "8x576 vs 96x1024" might mean:
    # - Input matrix: [8, 576]
    # - Weight matrix: [96, 1024] (which should be [1024, 576])
    
    # Let's check the matrix multiplication
    print(f"\nMatrix multiplication analysis:")
    input_tensor = torch.randn(8, 576)
    weight_tensor = torch.randn(1024, 576)  # Correct weight shape
    wrong_weight = torch.randn(96, 1024)    # Wrong weight shape from error
    
    print(f"Input: {input_tensor.shape}")
    print(f"Correct weight: {weight_tensor.shape}")
    print(f"Wrong weight: {wrong_weight.shape}")
    
    # Correct multiplication
    try:
        correct_output = torch.matmul(input_tensor, weight_tensor.t())
        print(f"Correct output: {correct_output.shape}")
    except Exception as e:
        print(f"Correct multiplication failed: {e}")
    
    # Wrong multiplication
    try:
        wrong_output = torch.matmul(input_tensor, wrong_weight)
        print(f"Wrong output: {wrong_output.shape}")
    except Exception as e:
        print(f"Wrong multiplication failed: {e}")
    
    # The issue is likely in the linear layer configuration
    print(f"\n=== CONCLUSION ===")
    print(f"The error '8x576 vs 96x1024' suggests:")
    print(f"1. Input tensor: [8, 576] - batch_size=8, patch_dim=576")
    print(f"2. Weight tensor expected: [1024, 576] but got [96, 1024]")
    print(f"3. patch_dim=576 = 16 * 6 * 6 = patch_len * n_features * n_tf")
    print(f"4. This indicates the patch_dim is incorrectly calculated including all timeframes")
    print(f"5. The linear layer is configured with wrong dimensions")

if __name__ == "__main__":
    debug_exact_issue()