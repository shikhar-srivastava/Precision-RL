"""Precision utilities for model training."""

import logging

import torch

logger = logging.getLogger(__name__)


def _make_norm_fp32_with_output_cast(target_dtype):
    """Create a forward hook that ensures norm outputs match model dtype.
    
    This hook allows norm weights to be in FP32 for better gradient precision
    while ensuring outputs are cast back to the model's base dtype (FP16/BF16)
    to maintain dtype consistency throughout the forward pass.
    
    The hook preserves gradients properly through the cast operation.
    """
    def forward_hook(module, input, output):
        # Handle different output types
        if isinstance(output, torch.Tensor):
            if output.dtype == torch.float32 and target_dtype != torch.float32:
                return output.to(target_dtype)
        elif isinstance(output, (tuple, list)):
            # Handle tuple/list outputs (some norms may return additional info)
            result = []
            for item in output:
                if isinstance(item, torch.Tensor) and item.dtype == torch.float32 and target_dtype != torch.float32:
                    result.append(item.to(target_dtype))
                else:
                    result.append(item)
            return type(output)(result)
        return output
    
    return forward_hook


def _cast_rmsnorm_weights_to_fp32(model, base_dtype=None) -> int:
    """Convert normalization weights to float32 in-place with output casting.

    Supports Qwen2RMSNorm (and other RMSNorm variants). Returns total number converted.

    **Why this is necessary:**
    Keeping norm weights in float32 improves gradient update precision when the
    model runs in reduced precision (bf16/fp16). Small weight updates near 1.0
    can underflow or be quantized away in low-precision formats.
    
    **How it works:**
    1. Cast norm layer weights to FP32 for better gradient precision
    2. Register forward hooks to cast outputs back to model dtype (FP16/BF16)
    3. This prevents dtype mismatches with subsequent layers
    
    **Why hooks are needed:**
    In RMSNorm's forward: `output = self.weight * normalized_hidden_states`
    When self.weight is FP32 and normalized_hidden_states is FP16, PyTorch's
    type promotion rules upcast the result to FP32. The hook casts it back.

    Args:
        model: The model whose RMSNorm weights should be cast to FP32.
        base_dtype: The base dtype of the model (e.g., torch.float16 or torch.bfloat16).
                   If None, will be inferred from the first Linear layer found.

    Returns:
        int: Total number of normalization layers converted to FP32.
    """
    rms_converted = 0
    rms_bias_converted = 0
    hooks_registered = 0
    
    # Infer base dtype if not provided
    if base_dtype is None:
        for module in model.modules():
            if isinstance(module, torch.nn.Linear) and hasattr(module, 'weight'):
                base_dtype = module.weight.dtype
                logger.info(f"Inferred base model dtype: {base_dtype}")
                break
        if base_dtype is None:
            logger.warning("Could not infer base dtype, defaulting to torch.float16")
            base_dtype = torch.float16
    
    # Skip if model is already in FP32
    if base_dtype == torch.float32:
        logger.info("Model is already in FP32, skipping norm weight casting")
        return 0

    for module_name, module in model.named_modules():
        try:
            # Handle Qwen2RMSNorm and other RMSNorm variants
            if "RMSNorm" in type(module).__name__:
                weight_converted = False
                if hasattr(module, "weight") and module.weight is not None and module.weight.dtype != torch.float32:
                    module.weight.data = module.weight.data.to(torch.float32)
                    rms_converted += 1
                    weight_converted = True
                    logger.debug(f"Converted {module_name} weight to float32")
                    
                if hasattr(module, "bias") and module.bias is not None and module.bias.dtype != torch.float32:
                    module.bias.data = module.bias.data.to(torch.float32)
                    rms_bias_converted += 1
                    logger.debug(f"Converted {module_name} bias to float32")
                
                # Register hook only if we converted the weight
                if weight_converted:
                    module.register_forward_hook(_make_norm_fp32_with_output_cast(base_dtype))
                    hooks_registered += 1
                    logger.debug(f"Registered output cast hook for {module_name}")
                    
            # Also handle LayerNorm for generality
            elif "LayerNorm" in type(module).__name__:
                weight_converted = False
                if hasattr(module, "weight") and module.weight is not None and module.weight.dtype != torch.float32:
                    module.weight.data = module.weight.data.to(torch.float32)
                    rms_converted += 1
                    weight_converted = True
                    logger.debug(f"Converted {module_name} weight to float32")
                    
                if hasattr(module, "bias") and module.bias is not None and module.bias.dtype != torch.float32:
                    module.bias.data = module.bias.data.to(torch.float32)
                    rms_bias_converted += 1
                    logger.debug(f"Converted {module_name} bias to float32")
                
                # Register hook only if we converted the weight
                if weight_converted:
                    module.register_forward_hook(_make_norm_fp32_with_output_cast(base_dtype))
                    hooks_registered += 1
                    logger.debug(f"Registered output cast hook for {module_name}")
                    
        except Exception as e:
            # Be conservative; skip any unexpected modules
            logger.warning(f"Failed to convert {module_name}: {e}")
            pass

    total_converted = rms_converted
    logger.info(
        f"Converted {total_converted} norm weights to float32 and {rms_bias_converted} biases"
    )
    logger.info(f"Registered {hooks_registered} forward hooks to cast outputs back to {base_dtype}")
    return total_converted


def _cast_norm_weights_for_sync(model, target_dtype):
    """Temporarily cast norm weights to target dtype for weight synchronization.
    
    This function is needed when syncing weights from learner (with FP32 norms)
    to actors/vLLM (expecting FP16/BF16). Call this before weight sync, then call
    _cast_rmsnorm_weights_to_fp32 again after sync to restore FP32 precision.
    
    Args:
        model: The model whose norm weights should be cast
        target_dtype: Target dtype (torch.float16 or torch.bfloat16)
    
    Returns:
        int: Number of norm weights cast
    """
    if target_dtype == torch.float32:
        return 0
    
    converted = 0
    for module in model.modules():
        if "RMSNorm" in type(module).__name__ or "LayerNorm" in type(module).__name__:
            if hasattr(module, "weight") and module.weight is not None:
                if module.weight.dtype != target_dtype:
                    module.weight.data = module.weight.data.to(target_dtype)
                    converted += 1
            if hasattr(module, "bias") and module.bias is not None:
                if module.bias.dtype != target_dtype:
                    module.bias.data = module.bias.data.to(target_dtype)
    
    logger.debug(f"Cast {converted} norm weights to {target_dtype} for weight sync")
    return converted
