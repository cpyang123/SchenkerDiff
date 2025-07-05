#!/usr/bin/env python3

import torch
import torch.nn as nn
from src.models.transformer_model import GraphTransformer

# Test script to reproduce and fix the GraphTransformer initialization error

def test_transformer_with_missing_args():
    """This will reproduce the error"""
    print("Testing GraphTransformer without required arguments...")
    
    # Define typical dimensions for testing
    input_dims = {'X': 10, 'E': 5, 'y': 3, 'r': 8}
    hidden_mlp_dims = {'X': 64, 'E': 32, 'y': 64, 'r': 32}
    hidden_dims = {'dx': 128, 'de': 64, 'dy': 128, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 128}
    output_dims = {'X': 10, 'E': 5, 'y': 3}
    
    try:
        # This should fail with the original error
        model = GraphTransformer(
            n_layers=3,
            input_dims=input_dims,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=output_dims
            # Missing act_fn_in and act_fn_out arguments
        )
        print("Unexpected success - model created without required arguments")
    except TypeError as e:
        print(f"Expected error caught: {e}")
        return e

def test_transformer_with_correct_args():
    """This should work correctly"""
    print("\nTesting GraphTransformer with correct arguments...")
    
    # Define typical dimensions for testing
    input_dims = {'X': 10, 'E': 5, 'y': 3, 'r': 8}
    hidden_mlp_dims = {'X': 64, 'E': 32, 'y': 64, 'r': 32}
    hidden_dims = {'dx': 128, 'de': 64, 'dy': 128, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 128}
    output_dims = {'X': 10, 'E': 5, 'y': 3}
    
    try:
        # This should work correctly
        model = GraphTransformer(
            n_layers=3,
            input_dims=input_dims,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=output_dims,
            act_fn_in=nn.ReLU(),  # Provide required argument
            act_fn_out=nn.ReLU()  # Provide required argument
        )
        print("Success! GraphTransformer created successfully with correct arguments")
        print(f"Model type: {type(model)}")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
        return model
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

if __name__ == "__main__":
    print("GraphTransformer Argument Test Script")
    print("=" * 50)
    
    # Test without required arguments (should fail)
    error = test_transformer_with_missing_args()
    
    # Test with correct arguments (should succeed)
    model = test_transformer_with_correct_args()
    
    if model is not None:
        print("\nAll tests completed successfully!")
    else:
        print("\nTest failed!")
