#!/bin/bash

# Change to the directory containing the Python script
cd /media/SSD2TB/fareed/wd/my_repos/tvm_8_bit/tvm-cuda-int8-benchmark
OPT_ITERATION=20

# Run the Python script
python3 tune_relay_cuda.py gprox_3 $OPT_ITERATION xgb_knob
python3 tune_relay_cuda.py gprox_3 $OPT_ITERATION random
python3 tune_relay_cuda.py gprox_3 $OPT_ITERATION ga

python3 tune_relay_cuda.py mob_v2 $OPT_ITERATION xgb_knob
python3 tune_relay_cuda.py mob_v2 $OPT_ITERATION random
python3 tune_relay_cuda.py mob_v2 $OPT_ITERATION ga

python3 tune_relay_cuda.py mob_v1 $OPT_ITERATION xgb_knob
python3 tune_relay_cuda.py mob_v1 $OPT_ITERATION random
python3 tune_relay_cuda.py mob_v1 $OPT_ITERATION ga

python3 tune_relay_cuda.py xce_r $OPT_ITERATION xgb_knob
python3 tune_relay_cuda.py xce_r $OPT_ITERATION random
python3 tune_relay_cuda.py xce_r $OPT_ITERATION ga
