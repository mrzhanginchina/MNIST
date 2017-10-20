import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#import mnist_inference
import leNet_5

# 配置神经网络中的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

#模型的保存的路径和文件名
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "model.ckpt"

def train(mnist):
    #定义输入和输出的placeholder
    #x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name="x-input")
    x = tf.placeholder(tf.float32, [BATCH_SIZE, leNet_5.IMAGE_SIZE, leNet_5.IMAGE_SIZE, leNet_5.NUM_CHANNELS],
                       name="x-input")

    y_ = tf.placeholder(tf.float32, [BATCH_SIZE, leNet_5.OUTPUT_NODE], name="y-output")

    regularize = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = leNet_5.inference(x, regularize, True)

    # 预测正确率的部分：
    # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage\
        (MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    # 这里的arg_max是在y_中选出最大值的下标, 1 代表是按照行来扫描.
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
        (logits=y, labels=tf.argmax(y_, 1))
    # 注意在这个地方, logits是我的神经网络中最后一层的具体输入, 而labels是我们的神经网络的期望输出.
    # logits代表的是logits的逻辑分类输出.
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))
    learning_rate = tf.train.exponential_decay\
        (LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE,
         LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).\
        minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train") #这句话不执行任何的操作,仅仅是一个占位符号.
        #但是在调用这句话之前,会执行with中的语句,这条语句同步更新了训练的参数和滑动平均.

    # 初始化TensorFlow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, [BATCH_SIZE, leNet_5.IMAGE_SIZE, leNet_5.IMAGE_SIZE, leNet_5.NUM_CHANNELS])
            _, loss_value, step= sess.run([train_op, loss, global_step],
                                           feed_dict={x: reshaped_xs, y_: ys})

            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g."
                      % (step, loss_value))
                # print("After %d training step(s), Accuracy on training batch is %g."
                #       % (step, acc))

        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
        print("保存模型成功！")


def main(argv=None):
    print("执行了Main函数")
    mnist = input_data.read_data_sets("./tem/data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    print("开始执行")
    tf.app.run()

