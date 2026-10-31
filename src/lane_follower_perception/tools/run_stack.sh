#!/bin/bash
# Start the whole driving stack against an already running Gazebo track world.
#
#   tools/run_stack.sh [scenario] [oncoming_speed] [use_tracker]
#     scenario        overtake (4 cars, default) | follow (one slow car on a bend)
#     oncoming_speed  m/s of the oncoming cars, default 0.3
#     use_tracker     true (default) | false  = overtake planner with / without measured speeds
#
# First, in another terminal:  ros2 launch lane_follower_perception track_phase2.launch.py drive:=false
# Logs of every node go to ~/ros2_ws/logs/. Ctrl+C stops all nodes started here.
SCEN=${1:-overtake}
ONC=${2:-0.3}
TRK=${3:-true}
LOG=~/ros2_ws/logs
mkdir -p "$LOG"
source ~/ros2_ws/install/setup.bash
for n in road_driver multitask_inference track_traffic overtake_planner lidar_tracker; do pkill -f "$n"; done
sleep 1
P="--ros-args -p use_sim_time:=true"
ros2 run lane_follower_perception multitask_inference_node.py $P -p steering_topic:=/cnn/cmd_vel \
    -r /lane/curvature:=/cnn/lane/curvature -r /lane/offset:=/cnn/lane/offset -r /lane/heading:=/cnn/lane/heading \
    > "$LOG/multitask_inference.log" 2>&1 &
ros2 run lane_follower_perception track_traffic.py $P -p scenario:=$SCEN -p oncoming_speed:=$ONC > "$LOG/track_traffic.log" 2>&1 &
ros2 run lane_follower_perception road_driver.py $P -p max_speed:=0.7 > "$LOG/road_driver.log" 2>&1 &
ros2 run lane_follower_perception lidar_tracker.py $P > "$LOG/lidar_tracker.log" 2>&1 &
ros2 run lane_follower_perception overtake_planner.py $P -p use_tracker:=$TRK > "$LOG/overtake_planner.log" 2>&1 &
stop() {
    echo; echo "stopping..."
    kill $(jobs -p) 2>/dev/null
    for n in road_driver multitask_inference track_traffic overtake_planner lidar_tracker; do pkill -f "$n"; done
    echo "overtakes completed: $(grep -c 'complete' "$LOG/overtake_planner.log")"
    echo "collisions: $(grep -c 'COLLISION' "$LOG/track_traffic.log")"
    exit 0
}
trap stop INT TERM
echo "✅ stack running: scenario=$SCEN, oncoming=$ONC m/s, tracker=$TRK. Logs in $LOG. Ctrl+C to stop."
sleep 2
tail -n +1 -f "$LOG/overtake_planner.log" | grep --line-buffered -E "\->|complete"
