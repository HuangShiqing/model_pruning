import tensorflow as tf
import numpy as np
import time
from data import read_data, data_generator
from model import model
from varible import *

x_train, y_train, x_valid, y_valid, x_test, y_test = read_data('/home/hsq/DeepLearning/data/dogVscat/train', 0.3, 0,
                                                                   pos_path="/dog/", neg_path="/cat/")

input_pb = tf.placeholder(tf.float32, [None, 224, 224, 3])
label_pb = tf.placeholder(tf.int32, [None])
logist, net = model(input_pb, is_train=False)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    try:
        saver.restore(sess, '/media/hsq/新加卷/ubuntu/ckpt/vgg16/4/' + "ep164-step45000-loss0.001")
        print("load ok!")
    except:
        print("ckpt文件不存在")
        raise

    data_yield = data_generator(x_valid, y_valid)
    error_num = 0
    i = 0
    for img, lable in data_yield:
        t1 = time.time()
        logist_out = sess.run(logist, feed_dict={input_pb: img})
        logist_out = np.argmax(logist_out, axis=-1)
        a = np.equal(logist_out, lable)
        a = list(a)
        error_num += a.count(False)
        i += 1
        # print(time.time()-t1)
    print('error: ' + str(error_num) + ' in ' + str(i * Gb_batch_size))
    exit()