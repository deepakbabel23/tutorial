import tensorflow as tf
print(tf.__version__)

tf.random.set_random_seed(5)
one = tf.random.uniform(shape=[1, 3], maxval=10, dtype=tf.int32, seed = 10)
two = tf.random.uniform(shape=[1, 3], maxval=10, dtype=tf.int32, seed = 10)

init_var = tf.global_variables_initializer()

with tf.Session() as sess1:
    sess1.run((init_var))
    sess1.run(one)
    print("one = ",one.eval())
    sess1.run(two)
    print("two = ",two.eval())
    sess1.run(one)
    print("one = ", one.eval())
    sess1.run(two)
    print("two = ", two.eval())

with tf.Session() as sess2:
    sess2.run((init_var))
    sess2.run(one)
    print("one = ",one.eval())
    sess2.run(two)
    print("two = ",two.eval())
    sess2.run(one)
    print("one = ", one.eval())
    sess2.run(two)
    print("two = ", two.eval())