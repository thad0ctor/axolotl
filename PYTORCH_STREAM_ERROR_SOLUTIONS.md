# PyTorch Stream Error Solutions

## Error: `opt_ready_stream && opt_parent_stream INTERNAL ASSERT FAILED`

This error indicates a PyTorch internal assertion failure in the autograd engine related to CUDA stream synchronization.

## Potential Solutions

### 1. Disable Activation Offloading
If you're using activation offloading with FSDP2, try disabling it:
```yaml
# In your config
fsdp2_activation_offloading: false
# or
activation_offloading: false
```

### 2. Switch to Legacy Activation Offloading
If you need activation offloading and are using LoRA:
```yaml
activation_offloading: "legacy"
```

### 3. Add CUDA Synchronization
Try adding explicit synchronization before training:
```python
# Add this before trainer.train()
import torch
if torch.cuda.is_available():
    torch.cuda.synchronize()
```

### 4. Environment Variables
Set these environment variables before running:
```bash
export CUDA_LAUNCH_BLOCKING=1  # Forces synchronous CUDA operations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # May help with memory fragmentation
```

### 5. Downgrade/Upgrade PyTorch
This appears to be a PyTorch bug. Try:
- Downgrading to PyTorch 2.3.x
- Upgrading to the latest PyTorch nightly build

### 6. Disable Mixed Precision Training
Temporarily disable mixed precision to isolate the issue:
```yaml
# In your config
bf16: false
fp16: false
```

### 7. Use FSDP1 Instead of FSDP2
If the issue persists with FSDP2:
```yaml
# Change from
fsdp2:
  sharding_strategy: FULL_SHARD
# To
fsdp:
  - full_shard
```

### 8. Check for LoRA + Custom Kernels
If using LoRA with custom kernels:
```yaml
# Disable custom kernels
lora_use_dora: false
# Or disable LoRA temporarily to test
```

### 9. Reduce Gradient Accumulation Steps
Try setting:
```yaml
gradient_accumulation_steps: 1
```

### 10. Report to PyTorch
Since this is an internal PyTorch error, consider:
1. Creating a minimal reproduction script
2. Reporting to PyTorch GitHub: https://github.com/pytorch/pytorch/issues

## Debugging Steps

1. **Check PyTorch Version**:
```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
```

2. **Monitor GPU Memory**:
```bash
nvidia-smi -l 1  # Monitor every second
```

3. **Enable Debugging**:
```bash
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
```

4. **Try Single GPU First**:
Test if the error occurs with single GPU to isolate distributed training issues.

## Related Issues
- PyTorch Issue #91241: Stream synchronization in autograd
- PyTorch Issue #97345: FSDP2 stream handling with mixed precision
- Axolotl: LoRA + activation offloading stream issues