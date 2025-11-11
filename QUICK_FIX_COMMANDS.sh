#!/bin/bash
# Quick Fix Commands for Scratch System
# Run these commands on the scratch system to deploy the fix

echo "=========================================="
echo "Deploying norm_in_fp32 Fix to Scratch"
echo "=========================================="

# Step 1: Copy precision.py to installed oat package
echo ""
echo "[1/3] Copying precision.py to oat package..."
cp /gpfs/fs2/scratch/ssrivas9/Precision-RL/oat/utils/precision.py \
   /scratch/ssrivas9/miniconda3/envs/oat2/lib/python3.10/site-packages/oat/utils/precision.py

if [ $? -eq 0 ]; then
    echo "✓ precision.py copied successfully"
else
    echo "✗ Failed to copy precision.py"
    exit 1
fi

# Step 2: Verify the fix is in place
echo ""
echo "[2/3] Verifying deployment..."

# Check for the new function
if grep -q "_cast_norm_weights_for_sync" /scratch/ssrivas9/miniconda3/envs/oat2/lib/python3.10/site-packages/oat/utils/precision.py; then
    echo "✓ _cast_norm_weights_for_sync function found"
else
    echo "✗ _cast_norm_weights_for_sync function NOT found"
    exit 1
fi

# Check for forward hook function
if grep -q "_make_norm_fp32_with_output_cast" /scratch/ssrivas9/miniconda3/envs/oat2/lib/python3.10/site-packages/oat/utils/precision.py; then
    echo "✓ Forward hook function found"
else
    echo "✗ Forward hook function NOT found"
    exit 1
fi

# Check main.py for sync_weights_to_actors
if grep -q "def sync_weights_to_actors" /gpfs/fs2/scratch/ssrivas9/Precision-RL/oat/main.py; then
    echo "✓ sync_weights_to_actors method found in main.py"
else
    echo "✗ sync_weights_to_actors method NOT found in main.py"
    echo "  Please ensure main.py is up to date"
    exit 1
fi

# Step 3: Ready to run
echo ""
echo "[3/3] Deployment complete!"
echo ""
echo "=========================================="
echo "✓ All checks passed!"
echo "=========================================="
echo ""
echo "You can now run your training scripts:"
echo ""
echo "  cd /gpfs/fs2/scratch/ssrivas9/Precision-RL/oat"
echo "  bash scripts/sanity/fp16_normfp32_grpo_4gpus.sh"
echo ""
echo "Or:"
echo ""
echo "  bash scripts/sanity/bf16_normfp32_grpo_4gpus.sh"
echo ""
echo "Expected behavior:"
echo "  ✓ No forward pass dtype errors"
echo "  ✓ No weight sync dtype errors"
echo "  ✓ Training proceeds normally"
echo ""
echo "=========================================="

