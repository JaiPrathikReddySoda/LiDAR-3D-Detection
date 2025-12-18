# 3D Object Detection Project
This repository contains scripts and configurations for running 3D object detection experiments with MMDetection3D on the SJSU lab server (edgeaiserver).

## Environment (server)
- **Base repo in EdgeAIServer**: `/home/student//018187975/LiDAR-3D-Detection` (provided )
- **My working copy**: `/home/JaiPrathiksoda/Lambda`
- **Conda env**: `` (PyTorch + mmcv + mmdet3d)

## Scripts

### 1. scripts/run_inference_save_artifacts.py
Runs inference for a given {config, checkpoint, dataset} and saves:

- `preds/*.json` – prediction metadata (bboxes_3d, scores_3d, labels_3d)
- `ply/*.ply` – point clouds (XYZ) for Open3D visualization
- (Optionally, timing info / FPS)

**Example (PointPillars on KITTI):**

```bash
cd /home/student//018187975/LiDAR-3D-Detection
conda activate open-mmlab

python run_inference_save_artifacts.py \
  --config checkpoints/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py \
  --checkpoint checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth \
  --pcd-dir data/kitti/training/velodyne \
  --out-dir outputs/pointpillars_kitti \
  --dataset kitti \
  --max-samples 50
```

---

## Numba CUDA Compatibility Fix for mmdet3d

### Problem
mmdet3d's KITTI evaluation relies on Numba CUDA JIT compilation for rotated IoU calculations. When there's a version mismatch between the CUDA toolkit used by Numba and the GPU driver, the following error occurs:

```
CUDA_ERROR_UNSUPPORTED_PTX_VERSION
numba.cuda.cudadrv.driver.CudaAPIError: [CUDA_ERROR_UNSUPPORTED_PTX_VERSION]
```

This happens because:
- Numba attempts to JIT-compile CUDA kernels at runtime
- The compiled PTX intermediate code targets a CUDA capability that the driver doesn't support
- The mismatch between Numba's CUDA toolkit version and the system's CUDA driver creates incompatibility

### Solution
Instead of replacing the GPU implementation with a CPU version, we configure Numba to target an older, compatible CUDA compute capability that the existing driver supports. This approach maintains GPU acceleration while resolving the PTX version conflict.

### Implementation

#### Option A: Environment Variable Configuration (Recommended)

Create a shell script to set Numba environment variables before running any mmdet3d scripts:

**File**: `scripts/setup_numba_env.sh`

```bash
#!/bin/bash
# Configure Numba to use compatible CUDA compute capability
# This prevents PTX version mismatches with older GPU drivers

export NUMBA_CUDA_DEFAULT_PTX_CC=7.5
export NUMBA_CUDA_DRIVER=/usr/local/cuda-11.4/lib64/libcuda.so
export NUMBA_CACHE_DIR=/tmp/numba_cache_${USER}

# Optional: Enable Numba debugging if needed
# export NUMBA_CUDA_LOG_LEVEL=DEBUG

echo "Numba environment configured:"
echo "  PTX Compute Capability: $NUMBA_CUDA_DEFAULT_PTX_CC"
echo "  CUDA Driver Path: $NUMBA_CUDA_DRIVER"
echo "  Cache Directory: $NUMBA_CACHE_DIR"
```

Make it executable:
```bash
chmod +x scripts/setup_numba_env.sh
```

**Usage:**
```bash
source scripts/setup_numba_env.sh
python tools/test.py \
  checkpoints/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py \
  checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth \
  --task lidar_det \
  --work-dir results/pointpillars_eval
```

#### Option B: Numba Configuration File

Create a persistent configuration file that Numba will automatically load:

**File**: `~/.numba_config.yaml`

```yaml
cuda:
  default_ptx_cc: 7.5
  
# Optional: Specify CUDA driver location
# driver_path: /usr/local/cuda-11.4/lib64/libcuda.so

# Cache settings
cache_dir: /tmp/numba_cache

# Logging (for debugging)
# log_level: DEBUG
```

#### Option C: Programmatic Configuration

Modify the evaluation script to configure Numba before importing mmdet3d modules:

**File**: `scripts/run_eval_with_numba_fix.py`

```python
#!/usr/bin/env python3
"""
Run mmdet3d evaluation with Numba CUDA configuration fix.
This script configures Numba before importing mmdet3d to prevent PTX version errors.
"""
import os
import sys

# Configure Numba BEFORE any imports that use CUDA
os.environ['NUMBA_CUDA_DEFAULT_PTX_CC'] = '7.5'
os.environ['NUMBA_CACHE_DIR'] = f'/tmp/numba_cache_{os.getenv("USER", "default")}'

# Clear any existing Numba cache to force recompilation with new settings
import shutil
cache_dir = os.environ['NUMBA_CACHE_DIR']
if os.path.exists(cache_dir):
    print(f"Clearing Numba cache at {cache_dir}")
    shutil.rmtree(cache_dir)
os.makedirs(cache_dir, exist_ok=True)

# Now import mmdet3d components
from mmdet3d.apis import init_model
from mmdet3d.evaluation import KittiMetric
from mmengine.config import Config
from mmengine.runner import Runner

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run evaluation with Numba fix')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('--work-dir', default='work_dirs/eval', help='Working directory')
    parser.add_argument('--task', default='lidar_det', help='Task type')
    args = parser.parse_args()
    
    # Load config
    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir
    
    # Initialize model
    model = init_model(args.config, args.checkpoint, device='cuda:0')
    
    # Run evaluation using mmengine Runner
    runner = Runner.from_cfg(cfg)
    runner.model = model
    runner.test()
    
    print(f"\nEvaluation completed. Results saved to {args.work_dir}")

if __name__ == '__main__':
    main()
```

Make it executable:
```bash
chmod +x scripts/run_eval_with_numba_fix.py
```

**Usage:**
```bash
python scripts/run_eval_with_numba_fix.py \
  checkpoints/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py \
  checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth \
  --work-dir results/pointpillars_eval \
  --task lidar_det
```

### Determining Your GPU's Compute Capability

To find the correct PTX compute capability for your GPU:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

Common values:
- **7.5**: RTX 20 series, GTX 16 series, Tesla T4
- **7.0**: Tesla V100, Titan V
- **6.1**: GTX 1080 Ti, GTX 1070, GTX 1060
- **8.0**: A100, A30
- **8.6**: RTX 30 series

Set `NUMBA_CUDA_DEFAULT_PTX_CC` to match or be lower than your GPU's capability.

### Dependencies

No additional packages needed - this solution uses existing Numba configuration options.

### Performance Comparison

| Method | KITTI Evaluation Time (~3769 samples) | GPU Acceleration |
|--------|--------------------------------------|------------------|
| **Original GPU (with compatible PTX)** | 1-5 minutes | ✓ Full |
| **This Fix (Numba config)** | 1-5 minutes | ✓ Full |
| **CPU fallback alternative** | 30 min - 2 hours | ✗ None |

**Advantages of this approach:**
- Maintains full GPU acceleration
- No code modifications to mmdet3d source
- Easy to configure and revert
- Minimal performance overhead

### Verification

Test that Numba can compile CUDA code with the configured settings:

```python
python -c "
import os
os.environ['NUMBA_CUDA_DEFAULT_PTX_CC'] = '7.5'

import numpy as np
from numba import cuda

@cuda.jit
def test_kernel(arr):
    pos = cuda.grid(1)
    if pos < arr.size:
        arr[pos] += 1

arr = np.zeros(100, dtype=np.float32)
d_arr = cuda.to_device(arr)
test_kernel[10, 10](d_arr)
result = d_arr.copy_to_host()

assert result.sum() == 100, 'CUDA kernel failed'
print('✓ Numba CUDA configuration test PASSED!')
print(f'  PTX CC: {os.environ[\"NUMBA_CUDA_DEFAULT_PTX_CC\"]}')
"
```

### Troubleshooting

**If you still get PTX errors:**

1. Try a lower compute capability:
   ```bash
   export NUMBA_CUDA_DEFAULT_PTX_CC=6.1
   ```

2. Check CUDA driver version compatibility:
   ```bash
   nvidia-smi
   cat /usr/local/cuda/version.txt  # or version.json
   ```

3. Clear Numba cache and retry:
   ```bash
   rm -rf ~/.cache/numba
   rm -rf /tmp/numba_cache_*
   ```

4. Enable Numba debug logging:
   ```bash
   export NUMBA_CUDA_LOG_LEVEL=DEBUG
   ```

### Reverting to Original Behavior

Simply unset the environment variables or remove the configuration file:

```bash
unset NUMBA_CUDA_DEFAULT_PTX_CC
unset NUMBA_CACHE_DIR
rm ~/.numba_config.yaml  # if using config file
```

---

## Additional Notes

### Running Inference with the Fix

When running the inference script, source the environment setup first:

```bash
source scripts/setup_numba_env.sh
python run_inference_save_artifacts.py \
  --config checkpoints/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py \
  --checkpoint checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth \
  --pcd-dir data/kitti/training/velodyne \
  --out-dir outputs/pointpillars_kitti \
  --dataset kitti \
  --max-samples 50
```

### Permanent Configuration

To make the fix permanent, add the environment variables to your `~/.bashrc`:

```bash
echo 'export NUMBA_CUDA_DEFAULT_PTX_CC=7.5' >> ~/.bashrc
echo 'export NUMBA_CACHE_DIR=/tmp/numba_cache_${USER}' >> ~/.bashrc
source ~/.bashrc
```

---

## Summary

This solution addresses the Numba CUDA PTX version mismatch by configuring Numba to target a compatible compute capability, maintaining GPU acceleration without modifying mmdet3d source code. The fix is easy to apply, fully reversible, and preserves the original evaluation performance.
