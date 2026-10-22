#!/usr/bin/env bash
# Reproducible WSL2 (Ubuntu 24.04) setup for pretrained PointPillars on nuScenes.
# Stack: Python 3.10 (uv-managed) / torch 2.1.2+cu121 / mmcv 2.1.0 (prebuilt CUDA ops)
#        / mmengine 0.10 / mmdet 3.3.0 / mmdet3d 1.4.0 / numpy<2. No CUDA Toolkit needed.
# Usage (inside WSL): bash pointpillars_nus/setup_wsl.sh
set -euo pipefail

if ! [ -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$HOME/pp/checkpoints" && cd "$HOME/pp"
[ -d .venv ] || uv venv --python 3.10 .venv
PY=.venv/bin/python

uv pip install --python $PY "numpy<2" \
  torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121 \
  --extra-index-url https://pypi.org/simple
uv pip install --python $PY mmcv==2.1.0 \
  --find-links https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html \
  --no-build-isolation
uv pip install --python $PY "numpy<2" "mmengine>=0.10,<1.0" mmdet==3.3.0 mmdet3d==1.4.0

URL=https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth
F="checkpoints/$(basename "$URL")"
[ -s "$F" ] || curl -fL --retry 3 -o "$F" "$URL"
sha256sum "$F" | grep -q '^fca299c1' || { echo "checkpoint hash mismatch"; exit 1; }

$PY -c "import torch, mmdet3d; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'mmdet3d', mmdet3d.__version__)"
