#!/usr/bin/env bash
# Model weights from the official sources, checked against weights.sha256.
#   conda activate conceptgraph && GSA_PATH=~/conceptgraph_ws/Grounded-Segment-Anything bash setup/download_weights.sh
set -e
HERE=$(dirname $(readlink -f $0)); G=${GSA_PATH:?set GSA_PATH}; cd $G; mkdir -p EfficientSAM
get() { [ -s "$2" ] || wget -q -c -O "$2.part" "$1" && { [ -s "$2" ] || mv "$2.part" "$2"; }; }
get https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth groundingdino_swint_ogc.pth
get https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth sam_vit_h_4b8939.pth
get https://huggingface.co/xinyu1205/recognize_anything_model/resolve/main/ram_swin_large_14m.pth ram_swin_large_14m.pth
get https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt EfficientSAM/mobile_sam.pt
sha256sum -c $HERE/weights.sha256
# CLIP ViT-H-14 (open_clip) and bert-base-uncased (GroundingDINO / RAM text encoder), same revisions as local
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('laion/CLIP-ViT-H-14-laion2B-s32B-b79K', revision='1c2b8495b28150b8a4922ee1c8edee224c284c0c',
                  allow_patterns=['open_clip_pytorch_model.bin'])
snapshot_download('bert-base-uncased', revision='86b5e0934494bd15c9632b12f734a8a67f723594',
                  allow_patterns=['config.json', 'model.safetensors', 'tokenizer_config.json', 'tokenizer.json', 'vocab.txt'])
print('HF models ready')
PY
echo "WEIGHTS DONE"
