cd /hnvme/workspace/v103fe17-lidar-bedlam || exit 1
squeue -u $USER -o "%.10i %.22j %.2t %.12M %.12L"
for j in $(squeue -u $USER -h -o "%i" -t R 2>/dev/null); do
  log=outputs/slurm-lidar-bedlam-$j.out
  [ -f "$log" ] || continue
  r=$(grep -m1 "^run " "$log" | awk '{print $2}')
  [ -n "$r" ] && echo "RUN $j $r"
done
