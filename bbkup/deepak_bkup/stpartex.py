from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.core.framework.node_def_pb2 import NodeDef
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
from tensorflow.python.ops import gen_stateful_random_ops
from tensorflow.python.ops import stateful_random_ops as random


import tensorflow as tf

sess1 = tf.InteractiveSession(config=config_pb2.ConfigProto(device_count={"CPU": 2}))
in_data1 = np.random.uniform(-5, 5, size=(3, 4, 5)).astype(np.float32)
var1 = tf.Variable(in_data1, name='in1')

@function.Defun(*[dtypes.float32] * 2)
def Body(x, y):
  # if x = 1, y = 2, ...
  # with ops.device("/job:localhost/replica:0/task:0/device:XLA_CPU:0"):
  with ops.device("/job:localhost/replica:0/task:0/device:CPU:0"):
    # a:= 1 + 1 = 2
    a = x + x
  with ops.device("/job:localhost/replica:0/task:0/device:CPU:0"):
    # b:= 2 + 2 = 4
    b = a + y
  return a + b

# output, = functional_ops.partitioned_call(args=[constant_op.constant(1.), constant_op.constant(2.)], f=Body)
op = tf.raw_ops.StatefulPartitionedCall(args=[constant_op.constant(1.), constant_op.constant(2.)], Tout=[dtypes.float32], f=Body)
print(type(op[0].op.node_def))
# op[0].op.device = "/job:localhost/replica:0/task:0/device:CPU:0"
op[0].op._is_stateful = True
print(op[0].op._is_stateful)
# tempOp = ops.Operation()
print (op[0].op)
print (type(op[0].op))
print(op)
# print(output)
print(sess1.run(op))
# if(sess1.run(output) == 6.):
#   print("success")
# else:
#   print("fail")