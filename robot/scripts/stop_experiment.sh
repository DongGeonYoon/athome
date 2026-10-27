#!/usr/bin/env bash
# Stop the perception run (CG-D, fusion node, rosbag). Localization and camera keep running (stop_all.sh stops them).
# The fusion node writes the final objects.json on SIGINT.
R=$(cat $HOME/real/results/CURRENT 2>/dev/null)
pkill -INT -f "ros2 bag recor[d]"; pkill -INT -f "lib/semantic_geometry_fusion/geometry_fusion_nod[e]"; pkill -INT -f "cgd_live_nod[e]"
for i in $(seq 30); do pgrep -f "geometry_fusion_nod[e]|ros2 bag recor[d]" >/dev/null || break; sleep 1; done
pkill -f "perception.launch.p[y]"
pkill -INT -f "monitor_run.p[y]"                       # writes monitor/report.md from the final objects.json
for i in $(seq 30); do pgrep -f "monitor_run.p[y]" >/dev/null || break; sleep 1; done
echo "[experiment] stopped. results: $R"; ls $R 2>/dev/null
[ -f "$R/objects.json" ] && python3 -c "import json,collections;o=json.load(open('$R/objects.json'))['objects'];print(len(o),'objects:',collections.Counter(x['semantic_tag'] for x in o).most_common(15))"
