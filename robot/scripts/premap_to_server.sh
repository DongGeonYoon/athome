#!/usr/bin/env bash
# Send a pre-mapping recording to the GPU server and build the Scene Graph dataset there (prebuild_from_bag.sh).
#   bash premap_to_server.sh NAME [--wait]      NAME = folder under ~/real/results (record_premap.sh)
#   --wait: wait for the server and copy the package back to ~/real/results/NAME/scene_graph/:
#     <VERSION>/ (scene_graph_dataset/, nav2_map/, pcd_map/, manifest.json, README.md) + NAME_<VERSION>.zip(.sha256) to send
# Server side: /data/athome/prebuild/robot_mapping/NAME/{bag, offline/}, log offline/prebuild.log
NAME=${1:?NAME}; R=$HOME/real/results/$NAME; A=/data/athome; D=$A/prebuild/robot_mapping/$NAME
source $HOME/real/config/gpu_server.env       # ATHOME_GPU_HOST / _PORT / _KEY
SSHC="ssh -o BatchMode=yes -i $ATHOME_GPU_KEY -p $ATHOME_GPU_PORT"; H=$ATHOME_GPU_HOST
RS() { rsync -a --partial -e "$SSHC" --rsync-path=$A/miniforge3/bin/rsync "$@"; }
[ -d $R/bag ] || { echo "no bag in $R"; exit 1; }
$SSHC $H "mkdir -p $D" && RS $R/bag $R/sync.json $R/start_pose.txt $R/calibration.yaml $H:$D/ && echo "[to_server] sent $(du -sh $R/bag | cut -f1)"
N0=$($SSHC $H "cat $D/offline/prebuild.log 2>/dev/null | wc -l")      # only log lines of this run count below
$SSHC $H "setsid nohup bash $A/server_sync/prebuild_from_bag.sh $NAME > /dev/null 2>&1 < /dev/null &" 
echo "[to_server] server processing started: $SSHC $H tail -f $D/offline/prebuild.log"
if [ "$2" = --wait ]; then
  until $SSHC $H "tail -n +$((N0 + 1)) $D/offline/prebuild.log | grep -q '\[prebuild\] done'" 2>/dev/null; do sleep 30; done
  V=$($SSHC $H "readlink $A/scene_graphs/$NAME/latest")
  mkdir -p $R/scene_graph && RS $H:$A/scene_graphs/$NAME/$V $H:$A/scene_graphs/$NAME/${NAME}_$V.zip $H:$A/scene_graphs/$NAME/${NAME}_$V.zip.sha256 $R/scene_graph/ \
    && (cd $R/scene_graph && sha256sum -c ${NAME}_$V.zip.sha256) && echo "[to_server] package $V -> $R/scene_graph/$V (+ ${NAME}_$V.zip)"
fi
