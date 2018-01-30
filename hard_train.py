import csv
import numpy as np
import tensorflow as tf
import tqdm


def from_csv(file_name: str) -> (np.ndarray, np.ndarray):
    """
    Load the data from specially formatted .csv file into list of tuples which contain vectors (input data
    and desired output).

    :param file_name: path to `.csv` file
    :return: tuple of X and Y numpy arrays
    """
    x, y = [], []
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            try:
                rx = list(map(lambda e: float(e), row[:row.index('-')]))
                ry = list(map(lambda e: float(e), row[row.index('-') + 1:]))
                x.append(rx)
                y.append(ry)
            except ValueError:
                pass
    x, y = np.array(x), np.array(y)
    return x, y


def separate_data(x: np.ndarray, y: np.ndarray, percentage: float) -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
    """
    Separate the data into training set and validation set with the given percentage

    :param x: list of vectors - input data
    :param y: list of vectors - desired output
    :param percentage: float value between [0, 1) which defines how much data move to validation set
    :return: tuple of training set, validation set
    """
    v_len = int(percentage * len(x))
    vx, vy = [], []
    tx, ty = [], []
    indices = np.random.choice(len(x), v_len)

    for i in range(len(x)):
        if i in indices:
            vx.append(x[i])
            vy.append(y[i])
        else:
            tx.append(x[i])
            ty.append(y[i])

    vx, vy = np.array(vx), np.array(vy)
    tx, ty = np.array(tx), np.array(ty)
    return tx, ty, vx, vy


def build_network(x, y):
    """
    Build Fully connected neural network

    :param x: `tf.placeholder` tensor
    :param y: `tf.placeholder` tensor
    :return: built neural network
    """
    nn = tf.layers.dense(x, x.shape[1], tf.nn.tanh)
    nn = tf.layers.dense(nn, 23, tf.nn.tanh)
    nn = tf.layers.dense(nn, 28, tf.nn.tanh)
    nn = tf.layers.dense(nn, 28, tf.nn.tanh)
    nn = tf.layers.dense(nn, 28, tf.nn.tanh)
    nn = tf.layers.dense(nn, 28, tf.nn.tanh)

    return tf.layers.dense(nn, y.shape[1], tf.nn.tanh, name='Output_layer')


if __name__ == '__main__':
    alpha = 0.7
    learn_rate = 0.05
    epochs = 1300
    batch_size = 8

    input_, output = from_csv("D:\\DELETE\\Дипломмо\\output.csv")
    tx, ty, vx, vy = separate_data(input_, output, 0.15)

    dim_x, dim_y = input_.shape[1], output.shape[1]

    x = tf.placeholder(shape=[None, dim_x], name='Times', dtype=tf.float32)
    y = tf.placeholder(shape=[None, dim_y], name='Marks', dtype=tf.float32)

    nn = build_network(x, y)

    reduction = tf.reduce_mean((y - nn) ** 2)
    loss = reduction + alpha * tf.nn.l2_loss(nn)
    optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for _ in tqdm.tqdm(range(epochs)):
            for i in range(0, len(tx), batch_size):
                _, err = sess.run([optimizer, loss],
                                  feed_dict={x: tx[i:i + batch_size], y: ty[i:i + batch_size]})

