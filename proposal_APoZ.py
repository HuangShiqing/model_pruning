import tensorflow as tf
import numpy as np
import os

from data import *
from model import *


def get_target_output(name, net):
    all = net.all_layers
    for i in range(len(all)):
        if name in all[i].name:
            return all[i]
    return None


def APoZ_proposal(n_filter, load_ckpt_dir, load_ckpt_name):
    # load_ckpt_dir = './ckpt'
    # load_ckpt_name = 'ep006-step3000-loss5.232'
    # n_filter = {'11': 64, '12': 64, '21': 128, '22': 128, '31': 256, '32': 256, '33': 256, '41': 512, '42': 512,
    #             '43': 512, '51': 512, '52': 512, '53': 512}
    x_train, y_train, x_valid, y_valid = read_data(Gb_data_dir)
    input_pb = tf.placeholder(tf.float32, [None, 224, 224, 3])
    logist, net = vgg16_adjusted(input_pb, n_filter=n_filter, is_train=False)
    saver = tf.train.Saver(max_to_keep=1000)

    with tf.Session() as sess:
        try:
            saver.restore(sess, os.path.join(load_ckpt_dir, load_ckpt_name))
            print("load ok!")
        except:
            print("ckpt文件不存在")
            raise

        apoz = {'11_bn': [], '12_bn': [], '21_bn': [], '22_bn': [], '31_bn': [], '32_bn': [], '33_bn': [],
                '41_bn': [], '42_bn': [], '43_bn': [], '51_bn': [], '52_bn': [], '53_bn': []}
        for key in apoz:
            # TODO: 不应该用测试集
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
            # print(key)
        keys = ['11_bn', '12_bn', '21_bn', '22_bn', '31_bn', '32_bn', '33_bn', '41_bn', '42_bn', '43_bn', '51_bn',
                '52_bn', '53_bn']
        ranks = []
        for key in keys:
            L1 = apoz[key]
            add_array = np.full(shape=(512 - L1.shape[-1]), fill_value=100)
            L1 = np.hstack((L1, add_array))
            ranks.append(L1)
        ranks = np.array(ranks)
        # TODO:可以再继续添加
        min_col = np.where(ranks == np.max(ranks))
        min_layer = keys[min_col[0][0]][:2]
        min_chanel = min_col[1][0]

    return {min_layer: [min_chanel]}


if __name__ == '__main__':
    a = APoZ_proposal()
