import tensorflow as tf
import numpy as np
import tqdm
import matplotlib.pyplot as pl


x = np.linspace(-1, 2, 1000)
y = np.sin(x) * np.e ** -x + np.random.normal(size=len(x))

alpha = 0.9

with tf.Session() as sess:
    x_ = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='input')
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='output')

    bias = tf.Variable(tf.random_normal([1]), name='bias')
    w1 = tf.Variable(tf.random_normal([1]))
    w2 = tf.Variable(tf.random_normal([1]))
    w3 = tf.Variable(tf.random_normal([1]))

    model_output = bias + w1 * x_ + w2 * tf.pow(x_, 2) + w3 * tf.pow(x_, 3)
    loss = tf.reduce_mean(tf.pow(y_ - model_output, 2)) + alpha * tf.nn.l2_loss(w1 + w2 + bias)

    gd = tf.train.GradientDescentOptimizer(0.01)

    train_step = gd.minimize(loss)
    sess.run(tf.global_variables_initializer())

    epochs = 1000
    for i in tqdm.tqdm(range(epochs)):
        unk, err = sess.run([train_step, loss], feed_dict={x_: x.reshape((len(x), 1)), y_: y.reshape((len(y), 1))})

    pl.plot(x, y)
    pl.plot(x, sess.run(model_output, feed_dict={x_: x.reshape((len(x), 1))}))
    pl.show()