import tensorflow as tf
# tf.compat.v1.enable_eager_execution()

n4 = tf.constant("hello")
n5 = tf.constant("world")

n6 = n4 + n5
print(n6)

n1 = tf.constant(1)
n2 = tf.constant(2)

n3 = n1 + n2
ten = tf.Tensor(n6)
print(ten)
print(n3)