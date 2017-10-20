import time
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
#import mnist_inference
import leNet_5
import mnist_train

EVAL_INTERVAL_SECS = 10


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        IMAGE_SIZE = mnist.validation.num_examples
        x = tf.placeholder(tf.float32, [IMAGE_SIZE, leNet_5.IMAGE_SIZE, leNet_5.IMAGE_SIZE, leNet_5.NUM_CHANNELS],
                           name='x-input')
        y_ = tf.placeholder(tf.float32, [IMAGE_SIZE, leNet_5.OUTPUT_NODE],
                            name="y-output")

        y = leNet_5.inference(x, None, False)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.\
            ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)

        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        with tf.Session() as sess:
            # 这个函数的意义是找到在这个路径下的checkpoint文件，如果有就返回这个文件，如果没有就是Null
            ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
            # 使用函数ckpt.model_checkpoint_path这个路径是存在的。
            if ckpt and ckpt.model_checkpoint_path:
                # 找到这个路径下的ckpt文件，使用restore来加载，在这里，restore会自动的找到ckpt文件中最新的ckpt文件。
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 在这里使用restore函数，不需要关心具体的三个文件，仅仅需要关注名称即可。
                # xs= mnist.validation.images
                # ys= mnist.validation.labels
                validate_feed = {x: np.reshape(mnist.validation.images, (IMAGE_SIZE, 28, 28, 1)), y_: mnist.validation.labels}
                accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                print("After training step(s), validation accuracy = %g"
                      % accuracy_score)
            else:
                print("NO MODEL is found!")
                return

def main(argv=None):
    mnist = input_data.read_data_sets("./tmp/data", one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    tf.app.run()