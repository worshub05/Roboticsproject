#!/bin/bash
# Final experiments, unattended: for every run it restarts Gazebo, starts the stack, lets it drive
# for a fixed time, stops it like Ctrl+C, and saves the summary and all node logs.
# Measurement runs (E..) have no camera; the video takes (V..) record a chase-camera mp4.
#
#   tools/run_experiments.sh              smoke test, then every run (about 80 minutes)
#   tools/run_experiments.sh E15 E17 V5   only these runs
#   DRY=1 tools/run_experiments.sh        only print the plan
# Results: ~/ros2_ws/experiments/<date_time>/<run id>/  and a copy in $WIN_OUT (a Windows folder).
PKG=~/ros2_ws/src/lane_follower_perception
LOG=~/ros2_ws/logs
OUT=${OUT:-~/ros2_ws/experiments/$(date +%Y%m%d_%H%M)}
WIN_OUT=${WIN_OUT:-}
ST="bash $PKG/tools/run_stack.sh"

# id | seconds | video (0/1) | options in front | run_stack arguments | title
RUNS=(
"E01|150|0||overtake 0.3 true mpc|Overtaking with MPC steering"
"E02|150|0||overtake 0.3 true tracker|Overtaking with the tracker controller"
"E03|300|0||follow 0.3 true mpc off fault|Broken ACC, RSS off"
"E04|300|0||follow 0.3 true mpc on fault|Broken ACC, RSS on"
"E05|100|0||cutin 0.3 true mpc off fault off|Cut-in 0.40 m: no RSS, no AEB"
"E06|100|0||cutin 0.3 true mpc on fault off|Cut-in 0.40 m: RSS only"
"E07|100|0||cutin 0.3 true mpc on fault on|Cut-in 0.40 m: RSS + AEB"
"E08|100|0|POPUP=0.25|cutin 0.3 true mpc on fault off|Cut-in 0.25 m: RSS only"
"E09|100|0|POPUP=0.25|cutin 0.3 true mpc on fault on|Cut-in 0.25 m: RSS + AEB"
"E10|300|0|V2I=off|lights 0.3 true mpc|Traffic lights without V2I"
"E11|300|0|GLOSA=off|lights 0.3 true mpc|Traffic lights: V2I stop only"
"E12|300|0||lights 0.3 true mpc|Traffic lights: V2I + GLOSA"
"E13|300|0|LOSS=0.3 DELAY=0.3|lights 0.3 true mpc|V2I + GLOSA, 30% lost + 0.3 s delay"
"E14|300|0|BLACKOUT=40.0|lights 0.3 true mpc|V2I blackout (fail-safe)"
"E15|130|0|PLATOON=acc|platoon 0.3 true mpc|Platoon: ACC, 1.0 s time gap"
"E16|130|0|PLATOON=acc_short|platoon 0.3 true mpc|Platoon: ACC, 0.3 s time gap"
"E17|130|0|PLATOON=cacc|platoon 0.3 true mpc|Platoon: CACC, 0.3 s time gap"
"E18|130|0|PLATOON=cacc LOSS=0.5 DELAY=0.2|platoon 0.3 true mpc|CACC, 50% lost + 0.2 s delay"
"E19|130|0|PLATOON=cacc BLACKOUT=40.0|platoon 0.3 true mpc|CACC, V2V blackout"
"V1|80|1||overtake 0.3 true mpc|Overtaking with MPC steering"
"V2|200|1||follow 0.3 true mpc on fault|RSS keeps a safe distance (broken ACC)"
"V3|70|1||cutin 0.3 true mpc on fault on|Emergency braking for a cut-in car"
"V4|110|1||lights 0.3 true mpc|Traffic lights with V2I + GLOSA"
"V5|80|1|PLATOON=cacc|platoon 0.3 true mpc|CACC platoon, 0.3 s time gap"
)
SMOKE="S0|30|1|PLATOON=cacc|platoon 0.3 true mpc|Smoke test"

source /opt/ros/humble/setup.bash 2>/dev/null
source ~/ros2_ws/install/setup.bash
mkdir -p "$OUT" "$LOG"
VIDEO_OK=1
python3 -c "import cv2" 2>/dev/null || { VIDEO_OK=0; echo "⚠ python cv2 not found: no videos will be recorded"; }

say() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/progress.txt"; [ -n "$WIN_OUT" ] && cp "$OUT/progress.txt" "$WIN_OUT/" 2>/dev/null; }

kill_all() {
    for n in road_driver multitask_inference track_traffic overtake_planner lidar_tracker rss_monitor aeb_node \
             traffic_light v2i_controller platoon.py cacc_controller energy_monitor follow_cam; do pkill -f "$n" 2>/dev/null; done
    [ -n "$GZ" ] && kill -TERM -- -"$GZ" 2>/dev/null
    sleep 2
    pkill -9 -f gzserver 2>/dev/null; pkill -9 -f gzclient 2>/dev/null
    [ -n "$GZ" ] && kill -9 -- -"$GZ" 2>/dev/null
    GZ=""
    sleep 2
}

start_gazebo() {     # $1 = log file. Returns 1 if the robot's /odom does not appear.
    kill_all
    setsid ros2 launch lane_follower_perception track_phase2.launch.py drive:=false > "$1" 2>&1 &
    GZ=$!
    for _ in $(seq 1 45); do
        sleep 2
        timeout 5 ros2 topic list 2>/dev/null | grep -q '^/odom$' && { sleep 8; return 0; }
    done
    return 1
}

run_one() {          # one line of RUNS
    IFS='|' read -r ID SECS VID ENVS ARGS TITLE <<< "$1"
    D="$OUT/$ID"
    mkdir -p "$D"
    echo "$ENVS run_stack.sh $ARGS   ($SECS s)   $TITLE" > "$D/command.txt"
    [ -n "$DRY" ] && { printf '%-4s %4s s  video=%s  %s run_stack.sh %s\n' "$ID" "$SECS" "$VID" "$ENVS" "$ARGS"; return 0; }
    say "▶ $ID: $TITLE ($SECS s)"
    touch "$D/.start"
    start_gazebo "$D/gazebo.log" || { say "❌ $ID: Gazebo did not start (no /odom)"; echo "$ID GAZEBO_FAILED" >> "$OUT/failed.txt"; return 1; }
    setsid env $ENVS $ST $ARGS > "$D/run.txt" 2>&1 &
    RS=$!
    CAM=""
    if [ "$VID" = "1" ] && [ "$VIDEO_OK" = "1" ]; then
        sleep 4
        python3 "$PKG/tools/follow_cam.py" --ros-args -p use_sim_time:=true -p out:="$D/$ID.mp4" -p title:="$TITLE" -p width:=640 -p height:=360 -p fps:=10 \
            > "$D/follow_cam.log" 2>&1 &
        CAM=$!
    fi
    sleep "$SECS"
    kill -TERM -- -"$RS" 2>/dev/null                   # stops it like Ctrl+C (run_stack prints its summary)
    for _ in $(seq 1 30); do kill -0 "$RS" 2>/dev/null || break; sleep 1; done
    kill -9 -- -"$RS" 2>/dev/null
    if [ -n "$CAM" ]; then
        kill -TERM "$CAM" 2>/dev/null
        for _ in $(seq 1 10); do kill -0 "$CAM" 2>/dev/null || break; sleep 1; done
        kill -9 "$CAM" 2>/dev/null
    fi
    find "$LOG" -maxdepth 1 -name "*.log" -newer "$D/.start" -exec cp {} "$D/" \; 2>/dev/null   # only this run's logs
    kill_all
    { echo "=== $ID: $TITLE"; echo "    $ENVS run_stack.sh $ARGS   ($SECS s)"; sed -n '/^stopping/,$p' "$D/run.txt" | tail -n +2; echo; } >> "$OUT/summary.txt"
    # a crash = a Traceback that is not just ROS complaining while the node is being stopped
    CRASHED=""
    for f in "$D"/*.log; do
        grep -q "Traceback" "$f" 2>/dev/null || continue
        grep -qE "ExternalShutdownException|context is invalid|context is not valid|rcl_shutdown already called|KeyboardInterrupt" "$f" && continue
        CRASHED="$CRASHED $(basename "$f")"
    done
    if [ -n "$CRASHED" ]; then
        say "⚠ $ID: a node crashed:$CRASHED"; echo "$ID CRASH$CRASHED" >> "$OUT/failed.txt"
    fi
    if [ -n "$WIN_OUT" ]; then mkdir -p "$WIN_OUT" && cp -r "$D" "$OUT/summary.txt" "$WIN_OUT/" 2>/dev/null; [ -f "$OUT/failed.txt" ] && cp "$OUT/failed.txt" "$WIN_OUT/"; fi
    say "✔ $ID done"
}

trap 'say "interrupted"; kill_all; exit 1' INT TERM

if [ -n "$DRY" ]; then
    for r in "${RUNS[@]}"; do run_one "$r"; done
    exit 0
fi

if [ $# -gt 0 ]; then
    for id in "$@"; do
        found=0
        for r in "${RUNS[@]}" "$SMOKE"; do [ "${r%%|*}" = "$id" ] && { run_one "$r"; found=1; }; done
        [ $found = 0 ] && say "unknown run id $id"
    done
else
    say "smoke test (30 s) before the long session"
    run_one "$SMOKE"
    grep -q "stack running" "$OUT/S0/run.txt" 2>/dev/null || { say "❌ smoke test: the stack did not start. Stopping."; exit 1; }
    grep -q "S0" "$OUT/failed.txt" 2>/dev/null && { say "❌ smoke test: a node crashed. Stopping."; exit 1; }
    if [ "$VIDEO_OK" = "1" ] && [ "$(stat -c %s "$OUT/S0/S0.mp4" 2>/dev/null || echo 0)" -lt 50000 ]; then
        say "⚠ smoke test: no video was recorded (see S0/follow_cam.log). Continuing with the measurement runs only."
        VIDEO_OK=0
    fi
    for r in "${RUNS[@]}"; do
        [ "$VIDEO_OK" = "0" ] && [ "$(echo "$r" | cut -d'|' -f3)" = "1" ] && continue
        run_one "$r"
    done
fi
say "🏁 all runs finished. Results in $OUT"
