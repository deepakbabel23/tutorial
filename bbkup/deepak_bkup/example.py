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

def _OptimizerOptions():
  for cse in [False, True]:
    for inline in [False, True]:
      for cfold in [False, True]:
        cfg = config_pb2.ConfigProto(
            graph_options=config_pb2.GraphOptions(
                optimizer_options=config_pb2.OptimizerOptions(
                    opt_level=config_pb2.OptimizerOptions.L0,
                    do_common_subexpression_elimination=cse,
                    do_function_inlining=inline,
                    do_constant_folding=cfold)))
        if cse:
          cfg.graph_options.rewrite_options.arithmetic_optimization = (
              rewriter_config_pb2.RewriterConfig.ON)
        else:
          cfg.graph_options.rewrite_options.arithmetic_optimization = (
              rewriter_config_pb2.RewriterConfig.OFF)
        if inline:
          cfg.graph_options.rewrite_options.function_optimization = (
              rewriter_config_pb2.RewriterConfig.ON)
        else:
          cfg.graph_options.rewrite_options.function_optimization = (
              rewriter_config_pb2.RewriterConfig.OFF)
        if cfold:
          cfg.graph_options.rewrite_options.constant_folding = (
              rewriter_config_pb2.RewriterConfig.ON)
        else:
          cfg.graph_options.rewrite_options.constant_folding = (
              rewriter_config_pb2.RewriterConfig.OFF)
        yield cfg

def my_py_func(x):
  x = tf.matmul(x, x)  # You can use tf ops
  # y = tf.constant(2,dtype=tf.resource)
  # g1 = tf.random.Generator.from_seed(1)
  # tf.raw_ops.StatefulUniformFullInt(resource=y,algorithm=g1,shape=(2,2),dtype=tf.int64,name=None)
  # tf.random_normal(shape=[4])
  # tf.Variable(use_resource=True)
  return x

with tf.Session() as sess:
  x = tf.placeholder(dtype=tf.float32)
  # Call eager function in graph!
  pf = tf.py_func(my_py_func, [x], tf.float32)

  sess.run(pf, feed_dict={x: [[2.0]]})  # [[4.0]]

class DevicePlacementTest(test.TestCase):

  def testNoDeviceGraph(self):
    with ops.Graph().as_default():

      @function.Defun(*[dtypes.float32] * 2)
      def Matmul(a, b):
        return math_ops.matmul(a, b)

      Matmul(1., 2.)

      gdef = ops.get_default_graph().as_graph_def()
      self.assertAllEqual(len(gdef.library.function), 1)
      fdef = gdef.library.function[0]

      for node in fdef.node_def:
        self.assertAllEqual(node.device, "")

  def testNestedDevices(self):
    with ops.Graph().as_default(), ops.device("CPU:0"):

      @function.Defun(*[dtypes.float32] * 2)
      def Matmul(a, b):
        return math_ops.matmul(a, b)

      with ops.device("CPU:1"):

        @function.Defun(*[dtypes.float32] * 2)
        def Divide(a, b):
          return math_ops.divide(a, b)

        Divide(Matmul(1., 2.), 3.)

      gdef = ops.get_default_graph().as_graph_def()
      matmul_fdef = [
          f for f in gdef.library.function if "Matmul" in f.signature.name
      ]
      divide_fdef = [
          f for f in gdef.library.function if "Divide" in f.signature.name
      ]
      self.assertAllEqual(len(matmul_fdef), 1)
      self.assertAllEqual(len(divide_fdef), 1)
      for node in matmul_fdef[0].node_def:
        self.assertAllEqual(node.device, "/device:CPU:0")
      for node in divide_fdef[0].node_def:
        self.assertAllEqual(node.device, "/device:CPU:1")

  def _testNestedDeviceWithSameFunction(self, func_name):

    def MatmulWrap(a, b):

      @function.Defun(
          func_name=func_name, *[dtypes.int32] * 2)
      def Matmul(a, b):
        return math_ops.matmul(a, b)

      return Matmul(a, b)

    with ops.Graph().as_default(), ops.device("CPU:0"):
      c = MatmulWrap(1, 2)

      with ops.device("CPU:1"):
        MatmulWrap(c, 3)

      gdef = ops.get_default_graph().as_graph_def()

      devices = []
      for node in gdef.library.function[0].node_def:
        devices.append(node.device)
      for node in gdef.library.function[1].node_def:
        devices.append(node.device)

      self.assertAllEqual(sorted(devices), ["/device:CPU:0", "/device:CPU:1"])

@test_util.run_deprecated_v1
def testFunctionMarkedStateful(self):

  @function.Defun(dtypes.int32, dtypes.float32)
  def Foo(t, x):
    return x[t]

  @function.Defun(dtypes.int64)
  def Bar(x):
    return x

  # NOTE(mrry): All functions are currently considered stateless by the
  # runtime, so we simulate a "stateful" function.
  # TODO(b/70565970): Remove this hack when we are able to build stateful
  # functions using the API.
  # pylint: disable=protected-access
  Foo._signature.is_stateful = True
  Bar._signature.is_stateful = True
  # pylint: enable=protected-access

  result_1 = Foo(3, [1.0, 2.0, 3.0, 4.0])
  result_2 = Bar(constant_op.constant(100, dtype=dtypes.int64))

  with session.Session() as sess:
    self.assertEqual(4.0, self.evaluate(result_1))
    self.assertEqual(100, self.evaluate(result_2))
    self.assertEqual((4.0, 100), sess.run((result_1, result_2)))

@test_util.run_deprecated_v1
def testStatefulFunction(self):

  @function.Defun()
  def FunctionWithStatelessOp():
    return constant_op.constant(42.0)

  @function.Defun()
  def FunctionWithStatefulOp():
    return random_ops.random_uniform([100], maxval=10, dtype=dtypes.int32)

  @function.Defun()
  def FunctionWithStatelessFunctionCall():
    return FunctionWithStatelessOp()

  @function.Defun()
  def FunctionWithStatefulFunctionCall():
    return FunctionWithStatefulOp()

  # Test that the `is_stateful` bit is propagated.
  self.assertFalse(FunctionWithStatelessOp.definition.signature.is_stateful)
  self.assertTrue(FunctionWithStatefulOp.definition.signature.is_stateful)
  self.assertFalse(
      FunctionWithStatelessFunctionCall.definition.signature.is_stateful)
  self.assertTrue(
      FunctionWithStatefulFunctionCall.definition.signature.is_stateful)

  # Ensure that two invocations of the same random-number-generating
  # function produce different results.
  result1 = FunctionWithStatefulFunctionCall()
  result2 = FunctionWithStatefulFunctionCall()

  # Statefulness affects how the function is treated by the various
  # optimization passes, so run the test in each optimizer
  # configuration.
  for config in _OptimizerOptions():
    with session.Session(config=config) as sess:
      val1, val2 = sess.run((result1, result2))
      self.assertFalse(all(val1 == val2))
      val3, val4 = sess.run((result1, result2))
      self.assertFalse(all(val3 == val1))
      self.assertFalse(all(val4 == val2))
devobj = DevicePlacementTest()
devobj.testNoDeviceGraph()
devobj.testNestedDevices()