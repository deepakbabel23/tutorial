import tensorflow as tf
import tensorflow_estimator as tfe
import os
import sys
import tensorflow as tf
from google.protobuf import text_format
from tensorflow.python.framework import graph_io
tf.debugging.set_log_device_placement(True)
print(tf.__version__)


sess1 = tf.InteractiveSession()
@tf.function
def add(a,b):
    return a+b

n1 = tf.constant(1)
n2 = tf.constant(2)

n3 = n1 + n2
print(n3)
print(sess1.run(n3))

with tf.device('/CPU:0'):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)


print(c)
print(sess1.run(c))

def load_pb(path_to_pb):
    with tf.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

def convert_pbtxt_to_pb(pbtxt):
    with open(pbtxt, 'r') as f:
      graph_def = tf.GraphDef()
      file_content = f.read()
      text_format.Merge(file_content, graph_def)
      graph_io.write_graph(graph_def,
              os.path.dirname(pbtxt),
              os.path.basename(pbtxt) + '.pb',
              as_text=False)
pbtxt = "/home/deepak/ofcprojects/amazon/tfrepo/tensorflow/graph-as-function.pbtxt"
convert_pbtxt_to_pb(pbtxt=pbtxt)

path = "/home/deepak/ofcprojects/amazon/tfrepo/tensorflow/graph-as-function.pbtxt.pb"
mygraph = load_pb(path)
print(mygraph)
# input = mygraph.get_tensor_by_name('input:0')
# output = mygraph.get_tensor_by_name('output')
sess1.run()
print("mygraph is :", mygraph)