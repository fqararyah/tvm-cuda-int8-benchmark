from tvm import relay
from tvm.relay import testing
from tvm.contrib import utils
import tvm
import tvm.contrib.graph_executor as runtime
import numpy as np
import time
import tvm.runtime.profiler_vm as profiler_vm

from tvm.contrib.debugger.debug_executor import GraphModuleDebug

import sys

MODEL_NAME = "gprox_3"
n_trial = 2
tuning_algo = "xgb_knob"

if len(sys.argv) > 1:
    MODEL_NAME = sys.argv[1]
if len(sys.argv) > 2:
    n_trial = int(sys.argv[2])
if len(sys.argv) > 3:
    tuning_algo = sys.argv[3]

path_lib = './builds_fp32/' + MODEL_NAME + '_' + tuning_algo + '_fp32_' + str(n_trial) +'.so'
originally_from_onnx_models = ['gprox_3']

def is_originally_from_onnx_model(model_name):
    for onnx_model_name in originally_from_onnx_models:
        if onnx_model_name in model_name:
            return True
    return False

# load it back
loaded_lib = tvm.runtime.load_module(path_lib)
print(loaded_lib.type_key)
print( type(loaded_lib.imported_modules[0]))

target = tvm.target.cuda()

dev = tvm.device(str(target), 0)
module = runtime.GraphModule(loaded_lib["default"](dev))

input_shape = (1, 3, 224, 224)

dtype = "float32"
if is_originally_from_onnx_model(MODEL_NAME):
    data_tvm = tvm.nd.array(
        (np.random.uniform(size=input_shape)).astype(dtype))
    module.set_input("input.1", data_tvm)
else:
    data_tvm = tvm.nd.array(
        (np.random.uniform(size=input_shape)).astype(dtype))
    module.set_input("input_1", data_tvm)

print(module.benchmark(dev, number=1, repeat=20))