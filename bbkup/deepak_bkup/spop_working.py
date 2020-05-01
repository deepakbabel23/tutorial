from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.framework import constant_op
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

import tensorflow as tf
sess1 = tf.InteractiveSession(config=config_pb2.ConfigProto(device_count={"CPU": 2}))

@function.Defun(*[dtypes.float32] * 2)
def FunctionWithStatefulOp(m,n):
    with ops.device("/job:localhost/replica:0/task:0/device:CPU:0"):
        mm = tf.Variable([1])
        x = random_ops.random_uniform([4], maxval=10.0, dtype=dtypes.float32)
    with ops.device("/job:localhost/replica:0/task:0/device:CPU:0"):
        y = random_ops.random_uniform([4], maxval=10.0, dtype=dtypes.float32)
    return x+y

ts = test.TestCase("runTest")
FunctionWithStatefulOp._signature.is_stateful = True
xx = tf.Variable([4], shape=(2,2))
yy = tf.Variable([4], shape=(2,2))
op1 = tf.raw_ops.StatefulPartitionedCall(args=[xx, yy], Tout=[dtypes.float32], f=FunctionWithStatefulOp)
op = tf.raw_ops.StatefulPartitionedCall(args=[constant_op.constant(1.), constant_op.constant(2.)], Tout=[dtypes.float32], f=FunctionWithStatefulOp)
# ts.assertFalse(FunctionWithStatefulOp.pdefinition.signature.is_stateful)
print(sess1.run(op))
print(op)