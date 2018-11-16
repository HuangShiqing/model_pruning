import tensorflow as tf
import numpy as np
import time
from data import read_data, data_generator
from model import model
from varible import *

from matplotlib import pyplot as plt


def get_target_variable(name):
    all = tf.global_variables()
    for i in range(len(all)):
        if name in all[i].name:
            return all[i]
    return None


f = open('./pruning_proposal/pruning_proposal_of_bn_53.txt', 'r')
proposal = eval(f.read())
f.close()

x_train, y_train, x_valid, y_valid, x_test, y_test = read_data('D:/DeepLearning/data2/dogVScat/train', 0.3, 0,
                                                               pos_path="/dog/", neg_path="/cat/")
input_pb = tf.placeholder(tf.float32, [None, 224, 224, 3])
label_pb = tf.placeholder(tf.int32, [None])
logist, net = model(input_pb, is_train=False, pruning=proposal)

varis = tf.global_variables()
var_to_restore = [val for val in varis if '53' not in val.name and 'fc3' not in val.name]
saver = tf.train.Saver(var_to_restore)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    try:
        saver.restore(sess, './ckpt/' + "ep164-step45000-loss0.001")
        print("load ok!")
    except:
        print("ckpt文件不存在")
        raise

    # 创建一个读写器
    reader = tf.train.NewCheckpointReader('./ckpt/' + "ep164-step45000-loss0.001")
    a = reader.get_tensor("conv_53/kernel")
    b = reader.get_tensor("conv_53/bias")
    c = reader.get_tensor("fc3_identity/W")
    d = reader.get_tensor("fc3_identity/b")

    e = proposal['bn_53']
    f = [i for i in range(a.shape[-1]) if i not in e]
    g = a[..., f]
    sess.run(tf.assign(get_target_variable('conv_53/kernel'), g))
    h = b[..., f]
    sess.run(tf.assign(get_target_variable('conv_53/bias'), h))
    j = c[f]
    sess.run(tf.assign(get_target_variable('fc3_identity/W'), j))

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
