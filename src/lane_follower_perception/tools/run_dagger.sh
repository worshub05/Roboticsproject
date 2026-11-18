#!/bin/bash
# DAgger, unattended: evaluate the current lane CNN, two rounds of collect (the network drives, the expert
# labels) + retrain on the aggregated data, then evaluate the new models. Gazebo is restarted for every
# drive. Safe to run again after an interruption: finished steps are skipped.
#
#   tools/run_dagger.sh          (about 2 hours: 9 evaluations, 4 collections, 2 trainings)
# Results: ~/ros2_ws/dagger/ (report.jsonl, logs, data) and a copy of the report in $WIN_OUT.
PKG=~/ros2_ws/src/lane_follower_perception
W=~/ros2_ws/dagger
MODELS=~/ros2_ws/models
BASE_DATA=${BASE_DATA:-~/ros2_ws/track_data_mt}
WIN_OUT=${WIN_OUT:-}
EVAL_S=${EVAL_S:-150}            # one evaluation drive (one lap of the oval is about 143 s)
N_COLLECT=${N_COLLECT:-1000}     # images per lane per round
EPOCHS=${EPOCHS:-4}
source /opt/ros/humble/setup.bash 2>/dev/null
source ~/ros2_ws/install/setup.bash
mkdir -p "$W/logs"
REPORT="$W/report.jsonl"; touch "$REPORT"
[ -f "$MODELS/lane_mt_v1.pt" ] || { echo "❌ $MODELS/lane_mt_v1.pt not found"; exit 1; }
[ -f "$BASE_DATA/labels.csv" ] || { echo "❌ original training data not found in $BASE_DATA"; exit 1; }

say() { echo "$(date +%H:%M:%S) $*" | tee -a "$W/progress.txt"; sync_out; }
sync_out() { [ -n "$WIN_OUT" ] && mkdir -p "$WIN_OUT" && cp "$W/progress.txt" "$REPORT" "$WIN_OUT/" 2>/dev/null; }
kill_all() {
    pkill -f "dagger.py" 2>/dev/null
    [ -n "$GZ" ] && kill -TERM -- -"$GZ" 2>/dev/null
    sleep 2; pkill -9 -f gzserver 2>/dev/null; pkill -9 -f gzclient 2>/dev/null
    [ -n "$GZ" ] && kill -9 -- -"$GZ" 2>/dev/null
    GZ=""; sleep 2
}
trap 'say "interrupted"; kill_all; exit 1' INT TERM

start_gazebo() {     # $1 = lane, $2 = log
    kill_all
    if [ "$1" = "other" ]; then POSE="x:=20 y:=0.5 yaw:=3.1416"; else POSE=""; fi
    setsid ros2 launch lane_follower_perception track_phase2.launch.py drive:=false $POSE > "$2" 2>&1 &
    GZ=$!
    for _ in $(seq 1 45); do
        sleep 2
        timeout 5 ros2 topic list 2>/dev/null | grep -q '^/camera/image_raw$' && \
            timeout 5 ros2 topic list 2>/dev/null | grep -q '^/odom$' && { sleep 6; return 0; }
    done
    return 1
}

done_tag() { grep -q "\"tag\": \"$1\"" "$REPORT"; }

drive() {            # tag lane model beta count noise out_dir
    local TAG=$1 LANE=$2 MODEL=$3 BETA=$4 COUNT=$5 NOISE=$6 OUTD=${7:-$W/unused}
    if done_tag "$TAG"; then say "skip $TAG (done)"; return 0; fi
    say "▶ $TAG: $( [ "$COUNT" -gt 0 ] && echo "collect $COUNT images" || echo "evaluate ${EVAL_S} s"), $(basename $MODEL), lane $LANE, beta $BETA, push $NOISE"
    start_gazebo "$LANE" "$W/logs/${TAG}_gazebo.log" || { say "❌ $TAG: Gazebo did not start"; return 1; }
    timeout $(( COUNT > 0 ? 900 : EVAL_S + 120 )) ros2 run lane_follower_perception dagger.py --ros-args \
        -p use_sim_time:=true -p tag:="$TAG" -p lane:="$LANE" -p model_path:="$MODEL" -p beta:="$BETA" \
        -p count:="$COUNT" -p duration:="$EVAL_S.0" -p noise_std:="$NOISE" -p out_dir:="$OUTD" \
        -p report:="$REPORT" > "$W/logs/$TAG.log" 2>&1
    kill_all
    if done_tag "$TAG"; then say "✔ $TAG: $(grep 'DAGGER RESULT' "$W/logs/$TAG.log" | tail -1 | sed 's/.*DAGGER RESULT: //')"
    else say "⚠ $TAG: no result (see logs/$TAG.log)"; return 1; fi
}

evaluate() {         # name model
    drive "eval_$1_ours" ours "$2" 0.0 0 0.0
    drive "eval_$1_other" other "$2" 0.0 0 0.0
    drive "eval_$1_push" ours "$2" 0.0 0 0.3
}

train() {            # out start data...
    local OUT=$1 START=$2; shift 2
    if [ -f "$OUT" ]; then say "skip training $(basename $OUT) (exists)"; return 0; fi
    say "▶ training $(basename $OUT) from $(basename $START) on: $*"
    python3 "$PKG/tools/dagger_train.py" --start "$START" --out "$OUT.tmp" --data "$@" --epochs "$EPOCHS" \
        > "$W/logs/train_$(basename $OUT .pt).log" 2>&1 && mv "$OUT.tmp" "$OUT" \
        && say "✔ $(tail -1 "$W/logs/train_$(basename $OUT .pt).log")" \
        || { say "❌ training failed (see logs/train_$(basename $OUT .pt).log)"; tail -5 "$W/logs/train_$(basename $OUT .pt).log"; exit 1; }
}

V1=$MODELS/lane_mt_v1.pt; D1=$MODELS/lane_mt_d1.pt; D2=$MODELS/lane_mt_d2.pt
say "DAgger session: evaluations of ${EVAL_S} s, ${N_COLLECT} images per lane per round, ${EPOCHS} epochs"
evaluate v1 "$V1"
drive round1_ours ours "$V1" 0.5 "$N_COLLECT" 0.0 "$W/round1"
drive round1_other other "$V1" 0.5 "$N_COLLECT" 0.0 "$W/round1"
train "$D1" "$V1" "$BASE_DATA" "$W/round1"
drive round2_ours ours "$D1" 0.0 "$N_COLLECT" 0.0 "$W/round2"
drive round2_other other "$D1" 0.0 "$N_COLLECT" 0.0 "$W/round2"
train "$D2" "$D1" "$BASE_DATA" "$W/round1" "$W/round2"
evaluate d1 "$D1"
evaluate d2 "$D2"
python3 "$PKG/tools/dagger_report.py" "$REPORT" | tee "$W/dagger_results.md"
[ -n "$WIN_OUT" ] && cp "$W/dagger_results.md" "$W"/logs/*.log "$WIN_OUT/" 2>/dev/null
say "🏁 DAgger finished. Table in $W/dagger_results.md"
