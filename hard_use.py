import numpy as np
import tensorflow as tf


if __name__ == '__main__':
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('model/network.meta')
        saver.restore(sess, 'model/network')

        vector = list(map(lambda e: int(e), input("Write vector (e.g. 10 12 0 0 3 17): ").split()))
        vector = np.array(vector).reshape((1, len(vector)))

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name('Times:0')
        nn = graph.get_tensor_by_name('Output_layer/Tanh:0')

        resp = sess.run(nn, feed_dict={x: vector})

        s = sum(resp[0])
        marks = [v/s * 50 for v in resp[0]]
        print(marks)
