from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import def_function as eager_def_function
from tensorflow.python.eager import function as eager_function
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
import tensorflow.python.ops.tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.util import compat

import tensorflow as tf
in_data1 = np.random.uniform(-5, 5, size=(3, 4, 5)).astype(np.float32)
var1 = tf.Variable(in_data1, name='in1')

@function.Defun(*[dtypes.float32] * 2)
def func1(x, y):
        return math_ops.multiply(x, y)


tensors = functional_ops.partitioned_call(
        args=[constant_op.constant(1.),
              constant_op.constant(2.)], f=func1)
print(tensors)

sess1 = tf.InteractiveSession(config=config_pb2.ConfigProto(device_count={"CPU": 2}))
config = config_pb2.ConfigProto(device_count={"CPU": 2})
@function.Defun(*[dtypes.float32] * 2)
def Body(x, y):
  # if x = 1, y = 2, ...
  with ops.device("/job:localhost/replica:0/task:0/device:CPU:0"):
    # a:= 1 + 1 = 2
    a = x + x
  with ops.device("/job:localhost/replica:0/task:0/device:CPU:1"):
    # b:= 2 + 2 = 4
    b = a + y
  return a + b
  # with ops.device("/cpu:2"):
  #   # c:= 2 + 4 = 6
  #   c = a + b
  # # a + b + c = 2 + 4 + 6 = 12
  # return a + b + c

# with sess(config=config_pb2.ConfigProto(device_count={"CPU": 1})):

@function.Defun(*[dtypes.float32] * 2)
def Body1(x, y):
  # if x = 1, y = 2, ...
  with ops.device("/job:localhost/replica:0/task:0/device:CPU:0"):
    # a:= 1 + 1 = 2
    a = x + x
  with ops.device("/job:localhost/replica:0/task:0/device:CPU:1"):
    # b:= 2 + 2 = 4
    b = a + y
  return a + b

partitionedOp, = functional_ops.partitioned_call(
  args=[constant_op.constant(1.),
        constant_op.constant(2.)], f=Body)
print(sess1.run(partitionedOp))

spop = gen_functional_ops.StatefulPartitionedCall(args=[constant_op.constant(1.), constant_op.constant(2.)], Tout=[dtypes.float32], f=Body1)

print(spop)
print(sess1.run(spop))


if(sess1.run(partitionedOp) == 6.):
  print("success")
else:
  print("fail")
