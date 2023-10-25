#!/bin/bash

# Change to the directory containing the Python script
cd /media/SSD2TB/fareed/wd/my_repos/tvm_8_bit/tvm-cuda-int8-benchmark
OPT_ITERATION=20

# Run the Python script

echo mob_v2 $OPT_ITERATION xgb_knob
python3 from_lib_fp32.py mob_v2 $OPT_ITERATION xgb_knob
echo mob_v2 $OPT_ITERATION random
python3 from_lib_fp32.py mob_v2 $OPT_ITERATION random
echo mob_v2 $OPT_ITERATION ga
python3 from_lib_fp32.py mob_v2 $OPT_ITERATION ga


echo mob_v1 $OPT_ITERATION xgb_knob
python3 from_lib_fp32.py mob_v1 $OPT_ITERATION xgb_knob
echo mob_v1 $OPT_ITERATION random
python3 from_lib_fp32.py mob_v1 $OPT_ITERATION random
echo mob_v1 $OPT_ITERATION ga
python3 from_lib_fp32.py mob_v1 $OPT_ITERATION ga


echo xce_r $OPT_ITERATION xgb_knob
python3 from_lib_fp32.py xce_r $OPT_ITERATION xgb_knob
echo xce_r $OPT_ITERATION random
python3 from_lib_fp32.py xce_r $OPT_ITERATION random
echo xce_r $OPT_ITERATION ga
python3 from_lib_fp32.py xce_r $OPT_ITERATION ga


echo gprox_3 $OPT_ITERATION xgb_knob
python3 from_lib_fp32.py gprox_3 $OPT_ITERATION xgb_knob
echo gprox_3 $OPT_ITERATION random
python3 from_lib_fp32.py gprox_3 $OPT_ITERATION random
echo gprox_3 $OPT_ITERATION ga
python3 from_lib_fp32.py gprox_3 $OPT_ITERATION ga

