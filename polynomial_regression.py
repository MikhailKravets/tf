import tensorflow as tf
import numpy as np
import tqdm
import matplotlib.pyplot as pl


def transform(x, power):
    res = np.hstack((np.ones(shape=(len(x), 1)), x))
    for p in range(2, power + 1):
        res = np.hstack((res, x ** p))
    return res


alpha = 0.2
order = 4
epochs = 4000

x = np.linspace(-1, 2, 1000)
y = np.sin(x) * np.e ** -x + np.random.normal(size=len(x), scale=0.5)

x_train = transform(x.reshape((len(x), 1)), order)
y_train = y.reshape((len(y), 1))

with tf.Session() as sess:
    x_ = tf.placeholder(dtype=tf.float32, shape=(None, order + 1), name='input')
    y_ = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='output')

    w = tf.Variable(tf.random_normal([order + 1, 1]), name='w')

    model_output = tf.matmul(x_, w)
    loss = tf.reduce_mean((y_ - model_output) ** 2) + alpha * tf.nn.l2_loss(w)

    gd = tf.train.GradientDescentOptimizer(0.01)

    train_step = gd.minimize(loss)
    sess.run(tf.global_variables_initializer())

    for i in tqdm.tqdm(range(epochs)):
        unk, err = sess.run([train_step, loss], feed_dict={x_: x_train, y_: y_train})

    pl.plot(x, y)
    pl.plot(x_train[:,1], sess.run(model_output, feed_dict={x_: x_train}))
    pl.show()