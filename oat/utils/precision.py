"""Precision utilities for model training."""

import logging

import torch

logger = logging.getLogger(__name__)


def _cast_rmsnorm_weights_to_fp32(model) -> int:
    """Convert normalization weights to float32 in-place.

    Supports Qwen2RMSNorm (and other RMSNorm variants). Returns total number converted.

    Keeping norm weights in float32 improves update precision when the
    model runs in reduced precision (bf16/fp16), because small updates from 1.0
    can underflow or be quantized away in low-precision formats.

    Args:
        model: The model whose RMSNorm weights should be cast to FP32.

    Returns:
        int: Total number of normalization layers converted to FP32.
    """
    rms_converted = 0
    rms_bias_converted = 0

    for module_name, module in model.named_modules():
        try:
            # Handle Qwen2RMSNorm and other RMSNorm variants
            if "RMSNorm" in type(module).__name__:
                if hasattr(module, "weight") and module.weight is not None and module.weight.dtype != torch.float32:
                    module.weight.data = module.weight.data.to(torch.float32)
                    rms_converted += 1
                    logger.debug(f"Converted {module_name} weight to float32")
                if hasattr(module, "bias") and module.bias is not None and module.bias.dtype != torch.float32:
                    module.bias.data = module.bias.data.to(torch.float32)
                    rms_bias_converted += 1
                    logger.debug(f"Converted {module_name} bias to float32")
            # Also handle LayerNorm for generality
            elif "LayerNorm" in type(module).__name__:
                if hasattr(module, "weight") and module.weight is not None and module.weight.dtype != torch.float32:
                    module.weight.data = module.weight.data.to(torch.float32)
                    rms_converted += 1
                    logger.debug(f"Converted {module_name} weight to float32")
                if hasattr(module, "bias") and module.bias is not None and module.bias.dtype != torch.float32:
                    module.bias.data = module.bias.data.to(torch.float32)
                    rms_bias_converted += 1
                    logger.debug(f"Converted {module_name} bias to float32")
        except Exception as e:
            # Be conservative; skip any unexpected modules
            logger.warning(f"Failed to convert {module_name}: {e}")
            pass

    total_converted = rms_converted
    logger.info(
        f"Converted {total_converted} norm weights to float32 and {rms_bias_converted} biases"
    )
    return total_converted

