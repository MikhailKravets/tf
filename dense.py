import tensorflow as tf
import numpy as np
import tqdm
import matplotlib.pyplot as pl


def f(x):
    return 0.5 * np.sin(np.exp(x)) - np.cos(np.exp(-1 * x))


alpha = 0.01
lrate = 0.05
order = 4
epochs = 50_000

x_real = np.linspace(-2.2, 2.5, 150)
x_train = x_real.copy()
np.random.shuffle(x_train)
y_train = f(x_train)
y_real = f(x_real)

x_train = x_train.reshape((len(x_train), 1))
x_real = x_real.reshape((len(x_real), 1))
y_train = y_train.reshape((len(y_train), 1))

x = tf.placeholder(shape=[None, 1], name='input', dtype=tf.float32)
y = tf.placeholder(shape=[None, 1], name='output', dtype=tf.float32)

nn = tf.layers.dense(x, 1, tf.nn.tanh)
nn = tf.layers.dense(nn, 10, tf.nn.tanh)
nn = tf.layers.dense(nn, 10, tf.nn.tanh)
nn = tf.layers.dense(nn, 10, tf.nn.tanh)
nn = tf.layers.dense(nn, 10, tf.nn.tanh)

out = tf.layers.dense(nn, 1, tf.nn.tanh, name='out_layer')

loss = tf.reduce_mean((y - out) ** 2) + alpha * tf.nn.l2_loss(out)
optimizer = tf.train.GradientDescentOptimizer(lrate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in tqdm.tqdm(range(epochs)):
        _, err = sess.run([optimizer, loss], feed_dict={x: x_train, y: y_train})

    pl.plot(x_real, y_real)
    pl.plot(x_real, sess.run(out, feed_dict={x: x_real}))
    pl.show()
