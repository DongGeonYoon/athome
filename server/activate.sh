# AtHOME GPU server environment. Everything (code, models, caches, conda) lives under /data/athome, the only
# path that survives a container restart. HOME is pointed there so the local scripts' $HOME/... paths work unchanged.
#   source /data/athome/activate.sh
export ATHOME=/data/athome HOME=/data/athome
export XDG_CACHE_HOME=$ATHOME/cache HF_HOME=$ATHOME/cache/huggingface TORCH_HOME=$ATHOME/cache/torch
export PIP_CACHE_DIR=$ATHOME/cache/pip CONDA_PKGS_DIRS=$ATHOME/cache/conda_pkgs TMPDIR=$ATHOME/tmp
export GSA_PATH=$ATHOME/conceptgraph_ws/Grounded-Segment-Anything PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
mkdir -p $TMPDIR
source $ATHOME/miniforge3/etc/profile.d/conda.sh && conda activate conceptgraph
cd $ATHOME
