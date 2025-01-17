# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Auto-tuning a Convolutional Network for NVIDIA GPU
==================================================
**Author**: `Lianmin Zheng <https://github.com/merrymercy>`_, `Eddie Yan <https://github.com/eqy/>`_

Auto-tuning for specific devices and workloads is critical for getting the
best performance. This is a tutorial on how to tune a whole convolutional
network for NVIDIA GPU.

The operator implementation for NVIDIA GPU in TVM is written in template form.
The template has many tunable knobs (tile factor, unrolling, etc).
We will tune all convolution and depthwise convolution operators
in the neural network. After tuning, we produce a log file which stores
the best knob values for all required operators. When the TVM compiler compiles
these operators, it will query this log file to get the best knob values.

We also released pre-tuned parameters for some NVIDIA GPUs. You can go to
`NVIDIA GPU Benchmark <https://github.com/apache/tvm/wiki/Benchmark#nvidia-gpu>`_
to see the results.

Note that this tutorial will not run on Windows or recent versions of macOS. To
get it to run, you will need to wrap the body of this tutorial in a :code:`if
__name__ == "__main__":` block.
"""

######################################################################
# Install dependencies
# --------------------
# To use the autotvm package in tvm, we need to install some extra dependencies.
# (change "3" to "2" if you use python2):
#
# .. code-block:: bash
#
#   pip3 install --user psutil xgboost tornado cloudpickle
#
# To make TVM run faster during tuning, it is recommended to use cython
# as FFI of tvm. In the root directory of tvm, execute:
#
# .. code-block:: bash
#
#   pip3 install --user cython
#   sudo make cython3
#
# Now return to python code. Import packages.

# sphinx_gallery_start_ignore
# sphinx_gallery_requires_cuda = True
# sphinx_gallery_end_ignore
import sys

import os

import numpy as np

import tvm
from tvm import relay, autotvm
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import tvm.contrib.graph_executor as runtime

import tensorflow.keras.applications as models
import tensorflow as tf
from tvm.contrib.download import download_testdata
import tvm.runtime.profiler_vm as profiler_vm

from prox_g.proxyless_nas import proxyless_net
import torch
import io
import onnx

#################################################################
# Define Network
# --------------
# First we need to define the network in relay frontend API.
# We can load some pre-defined network from :code:`tvm.relay.testing`.
# We can also load models from MXNet, ONNX and TensorFlow.

MODEL_NAME = "gprox_3"
n_trial = 2
tuning_algo = "xgb_knob"

if len(sys.argv) > 1:
    MODEL_NAME = sys.argv[1]
if len(sys.argv) > 2:
    n_trial = int(sys.argv[2])
if len(sys.argv) > 3:
    tuning_algo = sys.argv[3]

#### DEVICE CONFIG ####
target = tvm.target.cuda()

#### TUNING OPTION ####
log_file = "./logs/%s.log" % MODEL_NAME + '_' + tuning_algo + '_' + str(n_trial)
dtype = "float32"  # input just for the first layer before being quantized

tuning_option = {
    "log_filename": log_file,
    "tuner": tuning_algo,
    "n_trial": n_trial,
    "early_stopping": int(n_trial/2) - 1,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        runner=autotvm.LocalRunner(
            number=4, repeat=3, timeout=4, min_repeat_ms=150),
    ),
}

def from_torch(module, dummy_inputs):
    if isinstance(dummy_inputs, torch.Tensor):
        dummy_inputs = (dummy_inputs,)
    input_shape = {}
    for index, dummy_input in enumerate(dummy_inputs):
        if isinstance(dummy_input, np.ndarray):
            dummy_input = torch.from_numpy(dummy_input)
        input_shape['input.1'] = dummy_input.shape

    buffer = io.BytesIO()
    module.eval()
    torch.onnx.export(module, dummy_inputs, buffer)
    buffer.seek(0, 0)
    onnx_model = onnx.load_model(buffer)
    # for _input in onnx_model.graph.input:
    #     print(_input)
    return tvm.relay.frontend.from_onnx(onnx_model, shape=input_shape)

def get_network_keras_or_torch(batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    shape_dict = {"input_1": input_shape}

    if MODEL_NAME == 'resnet_50':
        model = model = models.ResNet50()
    elif MODEL_NAME == 'mob_v1':
        model = models.MobileNet()
    elif MODEL_NAME == 'mob_v1_0_5':
        model = models.MobileNet(alpha=0.5)
    elif MODEL_NAME == 'mob_v2':
        model = models.MobileNetV2()
    elif MODEL_NAME == 'mob_v2_0_5':
        model = models.MobileNetV2(alpha=0.5)
    elif MODEL_NAME == 'mob_v2_0_75':
        model = models.MobileNetV2(alpha=0.75)
    elif MODEL_NAME == 'mob_v2_0_25':
        model = models.MobileNetV2(alpha=0.35)
    elif MODEL_NAME == 'xce_r':
        model = models.Xception(input_shape=(224, 224, 3), weights=None)
    elif MODEL_NAME in ['gprox_3']:
        model = proxyless_net(2)

    if MODEL_NAME in ['gprox_3']:
        torch_input = torch.rand(1, 3, 224, 224)
        mod, params = from_torch(model, torch_input)
    else:
        mod, params = relay.frontend.from_keras(model, shape_dict)

    return mod, params, input_shape, output_shape


def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if "resnet" in name:
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer, batch_size=batch_size, dtype=dtype
        )
    elif "vgg" in name:
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.vgg.get_workload(
            num_layers=n_layer, batch_size=batch_size, dtype=dtype
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(
            batch_size=batch_size, dtype=dtype)
    elif name == "squeezenet_v1.1":
        mod, params = relay.testing.squeezenet.get_workload(
            batch_size=batch_size, version="1.1", dtype=dtype
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299)
        mod, params = relay.testing.inception_v3.get_workload(
            batch_size=batch_size, dtype=dtype)
    elif name == "mxnet":
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model

        block = get_model("resnet18_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(
            block, shape={"data": input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(
                net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape


###########################################
# Set Tuning Options
# ------------------
# Before tuning, we apply some configurations.

####################################################################
#
# .. note:: How to set tuning options
#
#   In general, the default value provided here works well.
#
#   If you have large time budget, you can set :code:`n_trial`, :code:`early_stopping` larger,
#   which makes the tuning runs longer.
#
#   If you have multiple devices, you can use all of them for measurement to
#   accelerate the tuning process. (see the 'Scale up measurement` section below).
#

###################################################################
# Begin Tuning
# ------------
# Now we can extract tuning tasks from the network and begin tuning.
# Here, we provide a simple utility function to tune a list of tasks.
# This function is just an initial implementation which tunes them in sequential order.
# We will introduce a more sophisticated tuning scheduler in the future.

# You can skip the implementation of this function for this tutorial.


def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=100,
    early_stopping=None,
    log_filename="tuning.log",
    use_transfer_learning=True,
):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb":
            tuner_obj = XGBTuner(tsk, loss_type="reg")
        elif tuner == "xgb_knob":
            tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="knob")
        elif tuner == "xgb_itervar":
            tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="itervar")
        elif tuner == "xgb_curve":
            tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="curve")
        elif tuner == "xgb_rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "xgb_rank_knob":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="knob")
        elif tuner == "xgb_rank_itervar":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="itervar")
        elif tuner == "xgb_rank_curve":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="curve")
        elif tuner == "xgb_rank_binary":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary")
        elif tuner == "xgb_rank_binary_knob":
            tuner_obj = XGBTuner(
                tsk, loss_type="rank-binary", feature_type="knob")
        elif tuner == "xgb_rank_binary_itervar":
            tuner_obj = XGBTuner(
                tsk, loss_type="rank-binary", feature_type="itervar")
        elif tuner == "xgb_rank_binary_curve":
            tuner_obj = XGBTuner(
                tsk, loss_type="rank-binary", feature_type="curve")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == "random":
            #print("rrrrrrrrrrrrrrrrrrrrr")
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(
                    autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.


def tune_and_evaluate(tuning_opt):
    # extract workloads from relay program
    print("Extract tasks...")
    mod, params, input_shape, out_shape = get_network_keras_or_torch(batch_size=1)
    # fareed
    with relay.quantize.qconfig(store_lowbit_output=False):
        mod = relay.quantize.quantize(mod, params=params)
    # end fareed
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params, ops=(
            relay.op.get("nn.conv2d"),)
    )
    # fareed
    # use int8 template_key
    for i in range(len(tasks)):
        tsk = tasks[i]
        if tsk.workload[0] != 'conv2d':
            continue
        input_channel = tsk.workload[2][1]
        output_channel = tsk.workload[2][0]
        if output_channel % 4 == 0 and input_channel % 4 == 0:
            tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                      tasks[i].target, tasks[i].target_host)
            tasks[i] = tsk
        # end fareed
    # ///////////////////tuning///////////////////
    # run tuning tasks

    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)
    # compile kernels with history best records
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=target, params=params)

        dev = tvm.device(str(target), 0)
        module = runtime.GraphModule(lib["default"](dev))
        data_tvm = tvm.nd.array(
            (np.random.uniform(size=input_shape)).astype(dtype))
        if MODEL_NAME in ['gprox_3']:
            module.set_input("input.1", data_tvm)
        else:
            module.set_input("input_1", data_tvm)

        lib.export_library('./builds/' + MODEL_NAME + '_' + tuning_algo + '_' + 
                        str(tuning_option['n_trial']) + '.so')

        # evaluate
        # 3
        # exe = relay.vm.compile(mod, target, params=params)
        # vm = profiler_vm.VirtualMachineProfiler(exe, dev)
        # report = vm.profile([data_tvm], func_name="main", number=1000, repeat=3)

        # with open('./logs/' + MODEL_NAME + '_' + str(n_trial) + '_profs.txt', 'w') as f:
        #     f.write(str(report))
        # 3
        # print("Evaluate inference time cost...")
        # print(module.benchmark(dev, number=1, repeat=1000))


tune_and_evaluate(tuning_option)
