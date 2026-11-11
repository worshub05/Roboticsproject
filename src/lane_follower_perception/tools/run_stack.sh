#!/bin/bash
# Start the whole driving stack against an already running Gazebo track world.
#
#   tools/run_stack.sh [scenario] [oncoming_speed] [use_tracker]
#     scenario        overtake (4 cars, default) | follow (one slow car on a bend)
#     oncoming_speed  m/s of the oncoming cars, default 0.3
#     use_tracker     true (default) | false  = overtake planner with / without measured speeds
#     controller      tracker (default) | mpc  = steering during peek / lane change / pass / return
#     rss             on (default) | off  = RSS speed limit enforced, or only monitored
#     aeb             on (default) | off  = 7th: emergency braking enforced, or only monitored
#     acc             normal (default) | fault  = fault injection: ACC keeps almost no gap (no overtaking)
#
# First, in another terminal:  ros2 launch lane_follower_perception track_phase2.launch.py drive:=false
# Logs of every node go to ~/ros2_ws/logs/. Ctrl+C stops all nodes started here.
SCEN=${1:-overtake}
ONC=${2:-0.3}
TRK=${3:-true}
CTRL=${4:-tracker}
RSS=${5:-on}
ACC=${6:-normal}
AEB=${7:-on}
AENF=true; [ "$AEB" = "off" ] && AENF=false
ENF=true; [ "$RSS" = "off" ] && ENF=false
ROADP="-p max_speed:=0.7"
PLANP=""
if [ "$ACC" = "fault" ]; then
    ROADP="$ROADP -p standstill_gap:=0.05 -p time_gap:=0.0 -p stop_distance:=0.05"
    PLANP="-p trigger_gap:=0.0"     # no overtaking: a pure car-following test
fi
[ "$SCEN" = "cutin" ] && PLANP="-p trigger_gap:=0.0"     # sudden-event test: no overtaking
# traffic lights (scenario lights). Options in front of the command: V2I=off, LOSS=0.3, DELAY=0.3, BLACKOUT=60
V2I=${V2I:-on}
GLOSA=${GLOSA:-on}
GLOSAP=true; [ "$GLOSA" = "off" ] && GLOSAP=false
# platoon (scenario platoon). Option in front of the command: PLATOON=acc | acc_short | cacc (default)
PLATOON=${PLATOON:-cacc}
if [ "$SCEN" = "platoon" ]; then
    PLANP="-p trigger_gap:=0.0"                     # follow the platoon, never overtake it
    [ "$PLATOON" = "acc_short" ] && ROADP="$ROADP -p time_gap:=0.3 -p standstill_gap:=0.4"
fi
LIGHTP="-p loss:=${LOSS:-0.0} -p delay:=${DELAY:-0.0} -p blackout:=${BLACKOUT:--1.0}"
LOG=~/ros2_ws/logs
mkdir -p "$LOG"
rm -f "$LOG/traffic_light.log" "$LOG/v2i.log" "$LOG/platoon.log" "$LOG/cacc.log"; touch "$LOG/traffic_light.log" "$LOG/v2i.log" "$LOG/platoon.log" "$LOG/cacc.log"
source ~/ros2_ws/install/setup.bash
for n in road_driver multitask_inference track_traffic overtake_planner lidar_tracker rss_monitor aeb_node traffic_light v2i_controller platoon.py cacc_controller; do pkill -f "$n"; done
sleep 1
P="--ros-args -p use_sim_time:=true"
ros2 run lane_follower_perception multitask_inference_node.py $P -p steering_topic:=/cnn/cmd_vel \
    -r /lane/curvature:=/cnn/lane/curvature -r /lane/offset:=/cnn/lane/offset -r /lane/heading:=/cnn/lane/heading \
    > "$LOG/multitask_inference.log" 2>&1 &
ros2 run lane_follower_perception track_traffic.py $P -p scenario:=$SCEN -p oncoming_speed:=$ONC -p popup_gap:=${POPUP:-0.4} > "$LOG/track_traffic.log" 2>&1 &
ros2 run lane_follower_perception road_driver.py $P $ROADP > "$LOG/road_driver.log" 2>&1 &
ros2 run lane_follower_perception lidar_tracker.py $P > "$LOG/lidar_tracker.log" 2>&1 &
if [ "$SCEN" = "lights" ]; then
    ros2 run lane_follower_perception traffic_light.py $P $LIGHTP > "$LOG/traffic_light.log" 2>&1 &
    [ "$V2I" = "on" ] && ros2 run lane_follower_perception v2i_controller.py $P -p glosa:=$GLOSAP > "$LOG/v2i.log" 2>&1 &
fi
if [ "$SCEN" = "platoon" ]; then
    ros2 run lane_follower_perception platoon.py $P -p mode:=$PLATOON -p loss:=${LOSS:-0.0} -p delay:=${DELAY:-0.0} -p blackout:=${BLACKOUT:--1.0} > "$LOG/platoon.log" 2>&1 &
    [ "$PLATOON" = "cacc" ] && ros2 run lane_follower_perception cacc_controller.py $P > "$LOG/cacc.log" 2>&1 &
fi
ros2 run lane_follower_perception aeb_node.py $P -p enforce:=$AENF > "$LOG/aeb.log" 2>&1 &
ros2 run lane_follower_perception rss_monitor.py $P -p enforce:=$ENF > "$LOG/rss_monitor.log" 2>&1 &
ros2 run lane_follower_perception overtake_planner.py $P -p use_tracker:=$TRK -p controller:=$CTRL $PLANP > "$LOG/overtake_planner.log" 2>&1 &
stop() {
    echo; echo "stopping..."
    kill $(jobs -p) 2>/dev/null
    timeout 5 ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{}" > /dev/null 2>&1
    for n in road_driver multitask_inference track_traffic overtake_planner lidar_tracker rss_monitor aeb_node traffic_light v2i_controller platoon.py cacc_controller; do pkill -f "$n"; done
    echo "overtakes completed: $(grep -c 'complete' "$LOG/overtake_planner.log")"
    echo "collisions: $(grep -c 'COLLISION' "$LOG/track_traffic.log")"
    grep 'TRAFFIC: robot' "$LOG/track_traffic.log" | tail -1 | sed 's/.*TRAFFIC: //'
    grep 'RSS STATS' "$LOG/rss_monitor.log" | tail -1 | sed 's/.*RSS STATS: /RSS: /'
    grep 'AEB STATS' "$LOG/aeb.log" | tail -1 | sed 's/.*AEB STATS: /AEB: /'
    if [ "$SCEN" = "platoon" ]; then
        grep 'braking event' "$LOG/platoon.log" | sed 's/.*PLATOON: //'
        grep 'PLATOON STATS' "$LOG/platoon.log" | tail -1 | sed 's/.*PLATOON STATS: /PLATOON: /'
        grep 'CACC STATS' "$LOG/cacc.log" | tail -1 | sed 's/.*CACC STATS: /CACC: /'
    fi
    [ "$SCEN" = "lights" ] && grep 'V2I STATS' "$LOG/traffic_light.log" | tail -1 | sed 's/.*V2I STATS: /LIGHTS: /'
    grep -E 'TRAFFIC: (cut-in|pop-up)' "$LOG/track_traffic.log" | sed 's/.*TRAFFIC: //'
    exit 0
}
trap stop INT TERM
echo "✅ stack running: scenario=$SCEN, oncoming=$ONC m/s, tracker=$TRK, controller=$CTRL, rss=$RSS, acc=$ACC, aeb=$AEB, v2i=$V2I, glosa=$GLOSA, platoon=$PLATOON. Logs in $LOG. Ctrl+C to stop."
sleep 2
tail -n +1 -f "$LOG/overtake_planner.log" "$LOG/rss_monitor.log" "$LOG/road_driver.log" "$LOG/aeb.log" "$LOG/track_traffic.log" "$LOG/traffic_light.log" "$LOG/v2i.log" "$LOG/platoon.log" "$LOG/cacc.log" | grep --line-buffered -E "BEHAVIOUR|complete|RSS:|AEB:|cut-in [0-9]|pop-up [0-9]|COLLISION|LIGHT:|V2I:|PLATOON:|CACC:|Traceback|Error"
