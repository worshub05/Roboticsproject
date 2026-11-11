#!/bin/bash
# The checks that GitHub Actions runs on every push (.github/workflows/tests.yml). No ROS, no Gazebo:
# only python3 + numpy, so the same command works on the laptop before pushing.
#   tools/ci.sh          1. every python file compiles   2. scenario suite (must be all PASS)
#   tools/ci.sh full     ... and 3. a short falsification search (report only, does not fail the run)
set -e -o pipefail
cd "$(dirname "$0")/.."
OUT=${CI_REPORT:-ci_report}
mkdir -p "$OUT"

echo "== 1. syntax: every python file compiles =="
python3 -m compileall -q lane_follower_perception tools
bash -n tools/run_stack.sh
echo "ok"
echo

echo "== 2. scenario suite =="
python3 tools/scenario_suite.py --json "$OUT/suite.json" | tee "$OUT/suite.txt"
echo

if [ "$1" = "full" ]; then
    echo "== 3. falsification (short search, report only) =="
    python3 tools/falsify.py --random 20 --refine 10 --json "$OUT/falsification.json" | tee "$OUT/falsification.txt"
    echo
fi
echo "✅ CI checks passed (reports in $OUT/)"
