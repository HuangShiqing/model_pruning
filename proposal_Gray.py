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


def Gray_proposal(n_filter, load_ckpt_dir, load_ckpt_name):
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

        gray = {'11_bn': [], '12_bn': [], '21_bn': [], '22_bn': [], '31_bn': [], '32_bn': [], '33_bn': [],
                '41_bn': [], '42_bn': [], '43_bn': [], '51_bn': [], '52_bn': [], '53_bn': []}
        for key in gray:
            # TODO: 不应该用测试集
            data_yield = data_generator(x_valid, y_valid)
            j = 1
            outs = None  # axis_0 不同图片序，axis_1 通道数
            for img, lable in data_yield:
                out = sess.run(get_target_output(key, net), feed_dict={input_pb: img})
                out = np.mean(out, axis=(1, 2))
                if j == 1:
                    outs = out
                else:
                    outs = np.concatenate((out, outs), axis=0)
                j += 1
                if j == 10:
                    break
            outs = outs.T
            p = 0.5
            avg_gray = list()
            temp = outs
            for index in range(len(outs)):
                outs = temp
                # 不同通道作为主序
                if index == 0:
                    outs = np.concatenate((np.expand_dims(outs[0], axis=0), outs[1:]), axis=0)
                elif index == len(outs) - 1:
                    outs = np.concatenate((np.expand_dims(outs[index], axis=0), outs[0:index]), axis=0)
                else:
                    outs = np.concatenate((np.expand_dims(outs[index], axis=0), outs[0:index], outs[index + 1:]), axis=0)
                # 2 无量纲化
                for i in range(len(outs)):
                    outs[i] = outs[i] / outs[i][0]  # max(outs[i])
                # 4 计算|x0-xi|
                for i in range(1, len(outs)):
                    outs[i] = abs(outs[i] - outs[0])
                outs = outs[1:]
                # 5 求最值
                min_num = np.min(outs)
                max_num = np.max(outs)
                # min_num = 0
                # max_num = 0
                # for i in range(1, len(outs)):
                #     min_temp = np.min(outs[i])
                #     max_temp = np.max(outs[i])
                #     if min_temp < min_num:
                #         min_num = min_temp
                #     if max_temp > max_num:
                #         max_num = max_temp
                # 6 计算关联系数
                for i in range(len(outs)):
                    outs[i] = (min_num + p * max_num) / (outs[i] + max_num * p)
                # 7 计算每个指标的关联度
                outs = np.mean(outs, axis=1)
                # 8 计算其他通道与参考序的平均关联度
                outs = np.mean(outs)
                avg_gray.append(outs)

            gray[key] = avg_gray
            print(key)
        keys = ['11_bn', '12_bn', '21_bn', '22_bn', '31_bn', '32_bn', '33_bn', '41_bn', '42_bn', '43_bn', '51_bn',
                '52_bn', '53_bn']
        ranks = []
        for key in keys:
            L1 = gray[key]
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
    n_filter = {'11': 64, '12': 64, '21': 128, '22': 128, '31': 256, '32': 256,
                '33': 256, '41': 512, '42': 512, '43': 512,
                '51': 512, '52': 512, '53': 512}
    load_dir = "/media/hsq/新加卷/ubuntu/ckpt/dogVScat/vgg16_adjusted/1/"
    load_name = "ep436-step273000-loss0.002"
    a = Gray_proposal(n_filter=n_filter, load_ckpt_dir=load_dir, load_ckpt_name=load_name)
    exit()
