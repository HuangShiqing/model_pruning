import tensorflow as tf
import numpy as np
import os

from model import *


def get_target_variable(name):
    all = tf.global_variables()
    for i in range(len(all)):
        if name in all[i].name:
            return all[i]
    return None


def L1_proposal(n_filter, load_ckpt_dir, load_ckpt_name):
    # load_ckpt_dir = './ckpt'
    # load_ckpt_name = 'ep006-step3000-loss5.232'
    # n_filter = {'11': 64, '12': 64, '21': 128, '22': 128, '31': 256, '32': 256, '33': 256, '41': 512, '42': 512,
    #             '43': 512,
    #             '51': 512, '52': 512, '53': 512}
    input_pb = tf.placeholder(tf.float32, [None, 224, 224, 3])
    logist, net = vgg16_adjusted(input_pb, n_filter=n_filter)
    saver = tf.train.Saver(max_to_keep=1000)

    with tf.Session() as sess:
        try:
            saver.restore(sess, os.path.join(load_ckpt_dir, load_ckpt_name))
            print("load ok!")
        except:
            print("ckpt文件不存在")
            raise

        ws = ['11_conv/kernel', '12_conv/kernel', '21_conv/kernel', '22_conv/kernel', '31_conv/kernel',
              '32_conv/kernel', '33_conv/kernel', '41_conv/kernel', '42_conv/kernel', '43_conv/kernel',
              '51_conv/kernel', '52_conv/kernel', '53_conv/kernel']
        ranks = []
        for w in ws:
            target_var = sess.run(get_target_variable(w))
            L1 = np.sum(np.abs(target_var), axis=(0, 1, 2))
            # TODO:添加Normalize
            add_array = np.full(shape=(512 - L1.shape[-1]), fill_value=100)
            L1 = np.hstack((L1, add_array))
            ranks.append(L1)
        ranks = np.array(ranks)
        # TODO:可以再继续添加
        min_col = np.where(ranks == np.min(ranks))
        min_layer = ws[min_col[0][0]][:2]
        min_chanel = min_col[1][0]
    return {min_layer: [min_chanel]}


if __name__ == '__main__':
    a = L1_proposal()
    exit()
