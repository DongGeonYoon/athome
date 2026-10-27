#!/usr/bin/env bash
# One-time server setup under /data/athome (run on the server after the code/models are synced):
#   bash /data/athome/server_sync/setup_server_env.sh
# miniforge -> /data/athome/miniforge3, env "conceptgraph" (python 3.10, torch 2.0.1 + cu118, same versions as the
# local PC), CG-D packages installed editable from the synced sources. Resumable: finished steps are skipped.
set -e
A=/data/athome
export HOME=$A XDG_CACHE_HOME=$A/cache PIP_CACHE_DIR=$A/cache/pip CONDA_PKGS_DIRS=$A/cache/conda_pkgs TMPDIR=$A/tmp
mkdir -p $TMPDIR $A/cache
if [ ! -x $A/miniforge3/bin/conda ]; then
  wget -q -O $TMPDIR/miniforge.sh https://github.com/conda-forge/miniforge/releases/download/24.11.3-2/Miniforge3-24.11.3-2-Linux-x86_64.sh
  bash $TMPDIR/miniforge.sh -b -p $A/miniforge3; rm $TMPDIR/miniforge.sh
fi
source $A/miniforge3/etc/profile.d/conda.sh
conda env list | grep -q "^conceptgraph " || conda create -y -n conceptgraph python=3.10.13
conda activate conceptgraph
python -c "import torch" 2>/dev/null || pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r $A/server_sync/requirements_server.txt
# local env has pillow 11.3 although supervision 0.14 asks for <10 (works in practice): same version, without the resolver
pip install --no-deps pillow==11.3.0
# no libGL in the container: open3d needs libGL.so.1 -> conda-forge libgl inside the env (not apt: outside /data)
[ -e $CONDA_PREFIX/lib/libGL.so.1 ] || conda install -y -c conda-forge libgl
# and OpenCV: headless build of the same OpenCV version
pip uninstall -y opencv-python opencv-python-headless; pip install --no-deps opencv-python-headless==4.8.0.76
C=$A/conceptgraph_ws; G=$C/Grounded-Segment-Anything
# same build tools as local (setuptools >= 81 has no pkg_resources, which torch 2.0 cpp_extension imports)
pip install setuptools==68.2.2 wheel==0.41.2
# GroundingDINO CUDA op: server toolkit is 11.3 but torch is cu118 -> CUDA 11.8 toolkit inside the env (under /data)
[ -x $CONDA_PREFIX/bin/nvcc ] || conda install -y -c nvidia/label/cuda-11.8.0 cuda-toolkit
export CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST="8.0"
pip install --no-deps -e $C/concept-graphs -e $G/segment_anything -e $G/recognize-anything
pip install --no-deps --no-build-isolation -e $G/GroundingDINO      # CUDA op is built if nvcc exists, else pure-python ops
pip install --no-deps $C/third_party/gradslam
python - <<'PY'
import torch, open_clip, groundingdino, segment_anything, ram, supervision
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
PY
echo "SETUP DONE"
