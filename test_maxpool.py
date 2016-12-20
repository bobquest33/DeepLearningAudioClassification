import tensorflow as tf


class TestMaxPool(tf.test.TestCase):
    def setUp(self):
        batch_size = 7
        channels = 1
        n_input = 4
        self.x = tf.placeholder(tf.float32, [None, n_input])
        k = 2
        self.input_tensor = tf.reshape(self.x, shape=[batch_size, 1, -1, channels])
        self.out = tf.nn.max_pool(self.input_tensor, ksize=[1, 1, k, 1], strides=[1, 1, k, 1],
                                  padding='SAME')

    def test_max_pooling(self):
        with self.test_session() as sess:
            input_data = [[100, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],
                          [1, 2, 3, 4]]
            print(sess.run(self.out, feed_dict={self.x: input_data}))
