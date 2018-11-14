import tensorflow as tf
import numpy as np
import time
from data import read_data, data_generator
from model import model
from varible import *

from matplotlib import pyplot as plt
import numpy as np


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


def plot_histogram(weights, image_name: str):
    """A function to plot weights distribution"""

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)

    ax.hist(weights,
            bins=100,
            facecolor='green',
            edgecolor='black',
            alpha=0.7,
            # range=(0.8, 1)
            )

    ax.set_title(image_name + '  APOZ Distribution')
    ax.set_ylabel('number of times')
    # ax.set_ylabel('Percentage')

    fig.savefig(image_name + '.png')


# 读取
f = open('temp.txt', 'r')
temp = f.read()
apoz = eval(temp)
f.close()

plot_histogram(apoz['bn_11'], 'bn_11')

mean_apoz = {}
for key in apoz:
    mean_apoz[key] = np.array(apoz[key]).mean()

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
    for key in apoz:
        data_yield = data_generator(x_valid, y_valid)
        j = 1
        for img, lable in data_yield:
            out = sess.run(get_target_output(key, net), feed_dict={input_pb: img})
            for i in range(out.shape[-1]):
                nonzero_num = np.count_nonzero(out[..., i])
                zero_num = (out[..., i].size - nonzero_num) / Gb_batch_size
                try:
                    apoz[key][i] = (apoz[key][i] * (j - 1) + zero_num) / j
                except:
                    apoz[key].append(zero_num)
            j += 1
        for i in range(len(apoz[key])):
            apoz[key][i] = apoz[key][i] / out[0, ..., 0].size
        print(key)
    # 保存
    f = open('temp.txt', 'w')
    f.write(str(apoz))
    f.close()

    # 读取
    # f = open('temp.txt', 'r')
    # a = f.read()
    # dict_name = eval(a)
    # f.close()
    exit()
