import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np

def np_mod(x,y):
    return (x % y).astype(np.float32)

def modgrad(op, grad):
    x = op.inputs[0] # the first argument (normally you need those to calculate the gradient, like the gradient of x^2 is 2x. )
    y = op.inputs[1] # the second argument

    return grad * 1, grad * tf.math.negative(tf.floordiv(x, y)) #the propagated gradient with respect to the first and second argument respectively

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):

    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def tf_mod(x,y, name=None):

    with ops.op_scope([x,y], name, "mod") as name:
        z = py_func(np_mod,
                        [x,y],
                        [tf.float32],
                        name=name,
                        grad=modgrad)  # <-- here's the call to the gradient
        return z[0]

with tf.Session() as sess:

    x = tf.constant([0.3,0.7,1.2,1.7])
    y = tf.constant([0.2,0.5,1.0,2.9])
    z = tf_mod(x,y)
    gr = tf.gradients(z, [x,y])
    tf.initialize_all_variables().run()
    x.eval()
    y.eval()
    z.eval()
    gr[0].eval()
    gr[1].eval()
    print(x.eval(), y.eval(),z.eval(), gr[0].eval(), gr[1].eval())