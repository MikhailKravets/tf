import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tqdm
import matplotlib.pyplot as pl


mnist = input_data.read_data_sets('tmp/mnist_data', one_hot=True)

x_train, y_train = mnist.train.images, mnist.train.labels

x = tf.placeholder(tf.float32, shape=[None, 784], name='x_place')
y = tf.placeholder(tf.float32, shape=[None, 10], name='y_place')


nn = tf.layers.conv2d(
    inputs=tf.reshape(x, shape=[-1, 28, 28, 1]),
    filters=32,
    kernel_size=(5, 5),
    padding='same',
    activation=tf.nn.relu
)

nn = tf.layers.max_pooling2d(
    inputs=nn,
    pool_size=(2, 2),
    strides=2,
)

nn = tf.layers.conv2d(
    inputs=nn,
    filters=64,
    kernel_size=(5, 5),
    padding='same',
    activation=tf.nn.relu
)

nn = tf.layers.max_pooling2d(
    inputs=nn,
    pool_size=(2, 2),
    strides=2
)

pool_dense = tf.layers.dense(tf.reshape(nn, [-1, 7 * 7 * 64]), units=1024, activation=tf.nn.relu)
res = tf.layers.dense(pool_dense, units=10, activation=tf.nn.sigmoid)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=res), name='Loss')
prediction = tf.argmax(res, axis=1)

optimizer = tf.train.GradientDescentOptimizer(0.03).minimize(loss)

epochs = 1200
errors = []
pred_error = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in tqdm.tqdm(range(epochs)):
        batch = mnist.train.next_batch(8)
        _, err = sess.run([optimizer, loss], feed_dict={x: batch[0].reshape(8, 784),
                                                        y: batch[1].reshape(8, 10)})
        errors.append(err)

    v = mnist.test.next_batch(1)
    pl.imshow(v[0].reshape(28, 28))
    print()
    print(f"Pred: {sess.run(prediction, feed_dict={x: v[0].reshape(1, 784)})}")
    print(f"Real: {v[1]}")

# pl.plot(errors)
pl.show()