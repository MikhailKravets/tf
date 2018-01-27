import tensorflow as tf
import numpy as np
import tqdm
import matplotlib.pyplot as pl


x = np.linspace(-5, 5, 1000)
y = np.sin(x) * np.e ** -x + np.random.normal(size=len(x))

sess = tf.Session()

x_ = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='input')
y_ = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='output')

model_output = tf.Variable(tf.random_normal([1]), name='bias') + tf.Variable(tf.random_normal([1])) * x_
loss = tf.reduce_mean(tf.pow(y_ - model_output, 2))

gd = tf.train.GradientDescentOptimizer(0.01)

train_step = gd.minimize(loss)
sess.run(tf.global_variables_initializer())

epochs = 100
for i in tqdm.tqdm(range(epochs)):
    unk, err = sess.run([train_step, loss], feed_dict={x_: x.reshape((len(x), 1)), y_: y.reshape((len(y), 1))})

pl.plot(x, y)
pl.plot(x, sess.run(model_output, feed_dict={x_: x.reshape((len(x), 1))}))
pl.show()