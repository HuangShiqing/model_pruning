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


def get_target_output(name, net):
    all = net.all_layers
    for i in range(len(all)):
        if name in all[i].name:
            return all[i]
    return None


x_train, y_train, x_valid, y_valid, x_test, y_test = read_data('/home/hsq/DeepLearning/data/dogVscat/train', 0.3, 0,
                                                               pos_path="/dog/", neg_path="/cat/")
input_pb = tf.placeholder(tf.float32, [None, 224, 224, 3])
label_pb = tf.placeholder(tf.int32, [None])
logist, net = model(input_pb, is_train=False)

saver = tf.train.Saver()
with tf.Session() as sess:
    try:
        saver.restore(sess, '/media/hsq/新加卷/ubuntu/ckpt/vgg16/4/' + "ep164-step45000-loss0.001")
        print("load ok!")
    except:
        print("ckpt文件不存在")
        raise

    apoz = {'bn_11': [], 'bn_12': [], 'bn_21': [], 'bn_22': [], 'bn_31': [], 'bn_32': [], 'bn_33': [], 'bn_41': [],
            'bn_42': [], 'bn_43': [], 'bn_51': [], 'bn_52': [], 'bn_53': []}

    data_yield = data_generator(x_valid, y_valid)
    j = 1
    for img, lable in data_yield:
        bn_11 = sess.run(get_target_output('bn_11', net), feed_dict={input_pb: img})

        for i in range(bn_11.shape[-1]):
            nonzero = np.count_nonzero(bn_11[..., i])
            zero_num = (bn_11[..., i].size - nonzero) / Gb_batch_size
            try:
                apoz['bn_11'][i] = (apoz['bn_11'][i] * (j - 1) + zero_num) / j
            except:
                apoz['bn_11'].append(zero_num)
        j += 1
    # apoz['bn_11'] = apoz['bn_11'] / bn_11[0, ..., 0].size
    exit()
