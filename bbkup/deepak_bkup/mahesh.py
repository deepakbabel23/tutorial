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
from tensorflow.python.ops import state_ops

import tensorflow as tf

#
# in_data1 = np.random.uniform(-5, 5, size=(3, 4, 5)).astype(np.float32)
# var1 = tf.Variable(in_data1, name='in1')
#
@function.Defun(*[dtypes.float32] * 2)
def func1(x, y):
        return math_ops.multiply(x, y)


# tensors = functional_ops.partitioned_call(
#         args=[constant_op.constant(1.),
#               constant_op.constant(2.)], f=func1)

spop = gen_functional_ops.StatefulPartitionedCall(args=[constant_op.constant(1.), constant_op.constant(2.)],Tout=[dtypes.float32],f=func1)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(spop))
# print(sess.run(tensors))
print("hi")
#
# def testPerStepTrace():
#     run_options = config_pb2.RunOptions(
#         trace_level=config_pb2.RunOptions.FULL_TRACE)
#     run_metadata = config_pb2.RunMetadata()
#
#     with ops.device('/cpu:0'):
#       with session.Session() as sess:
#         sess.run(constant_op.constant(1.0))
#         # self.assertTrue(not run_metadata.HasField('step_stats'))
#
#         sess.run(constant_op.constant(1.0), run_metadata=run_metadata)
#         # self.assertTrue(not run_metadata.HasField('step_stats'))
#
#         sess.run(constant_op.constant(1.0),
#                  options=run_options,
#                  run_metadata=run_metadata)
#         print(run_metadata)
#
#         # self.assertTrue(run_metadata.HasField('step_stats'))
#         # self.assertEquals(len(run_metadata.step_stats.dev_stats), 1)
def testExtendWithStatefulOperations():
    run_options = config_pb2.RunOptions(
        trace_level=config_pb2.RunOptions.FULL_TRACE)
    run_metadata = config_pb2.RunMetadata()
    with session.Session(config=config_pb2.ConfigProto(device_count={"CPU": 4})) as s:
        with ops.device('/cpu:0'):
            a = constant_op.constant(1.0, shape=[1, 2])
            b = constant_op.constant(2.0, shape=[2, 3])
            c = math_ops.matmul(a, b)
            v = variables.Variable(c, name='testExtendWithStatefulOperations_v')
            v.initializer.run()
            v_val = v.eval()
            # self.assertAllEqual([[4.0, 4.0, 4.0]], v_val)
        with ops.device('/cpu:1'):
            d = constant_op.constant(3.0, shape=[2, 3])
            e = math_ops.matmul(a, d)
            assign_e_to_v = state_ops.assign(v, e)
            # Extend will happen here.
            e_val = e.eval()
            # self.assertAllEqual([[6.0, 6.0, 6.0]], e_val)
            v_val = v.eval()
            # self.assertAllEqual([[4.0, 4.0, 4.0]], v_val)
            s.run(assign_e_to_v, options=run_options, run_metadata=run_metadata)
            print("hello")
            v_val = v.eval()
            # self.assertAllEqual([[6.0, 6.0, 6.0]], v_val)

# testPerStepTrace()
testExtendWithStatefulOperations()
