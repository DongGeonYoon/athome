#!/usr/bin/env bash
# Is every topic being recorded? Never opens the bag database while the recorder runs (reading it locked SQLite and
# killed the run3 recorder): during recording it uses the recorder log, file growth and topic rates; afterwards bag info.
R=$(cat $HOME/real/results/CURRENT 2>/dev/null); source $HOME/real/scripts/env.sh
echo "[check] $R/bag"
grep -iE "unknown type|error|exception" $R/logs/bag.log | sort -u | head -5
if pgrep -f "ros2 bag recor[d]" >/dev/null; then
  echo "recording: subscribed topics"; grep -oE "Subscribed to topic '[^']+'" $R/logs/bag.log | sed "s/Subscribed to topic /  /" | sort -u
  s1=$(du -sb $R/bag | cut -f1); sleep 3; s2=$(du -sb $R/bag | cut -f1); echo "bag growing: $(( (s2 - s1) / 3 / 1024 )) KB/s"
  for t in /cg_d/objects_2d /cg_d/annotated /athome/perception/objects /camera/camera/color/image_raw/compressed /cloud_registered_body; do
    echo "  $t publishers: $(ros2 topic info $t 2>/dev/null | grep -oP 'Publisher count: \K\d+')"; done
else
  ros2 bag info $R/bag 2>/dev/null | grep -E "Duration|Topic:" | sed 's/| Serialization Format: cdr//'
fi
echo "annotated jpg: $(ls $R/annotated 2>/dev/null | wc -l), detections.jsonl: $(wc -l < $R/detections.jsonl 2>/dev/null)"
