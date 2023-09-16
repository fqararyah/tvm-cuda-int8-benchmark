from tvm import relay
from tvm.relay import testing
from tvm.contrib import utils
import tvm
import tvm.contrib.graph_executor as runtime
import numpy as np
import time
import tvm.runtime.profiler_vm as profiler_vm

from tvm.contrib.debugger.debug_executor import GraphModuleDebug

MODEL_NAME = 'xce_r'
n_iters = 50
path_lib = './builds/' + MODEL_NAME + '_' + str(n_iters) + '.so'

# load it back
loaded_lib = tvm.runtime.load_module(path_lib)
print(loaded_lib.type_key)
print( type(loaded_lib.imported_modules[0]))

target = tvm.target.cuda()

dev = tvm.device(str(target), 0)
module = runtime.GraphModule(loaded_lib["default"](dev))

input_shape = (1, 3, 224, 224)

dtype = "float32"
data_tvm = tvm.nd.array(
    (np.random.uniform(size=input_shape)).astype(dtype))
module.set_input("input_1", data_tvm)

t1 = time.time()
module.run()
t2 = time.time()

print(module.benchmark(dev, number=1, repeat=1000))