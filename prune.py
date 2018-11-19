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


f = open('./pruning_proposal/pruning_proposal_of_bn_21_0.3.txt', 'r')
proposal = eval(f.read())
f.close()

x_train, y_train, x_valid, y_valid, x_test, y_test = read_data('/home/xinje/hsq/data/dogVScat', 0.3, 0,
                                                               pos_path="/dog/", neg_path="/cat/")
input_pb = tf.placeholder(tf.float32, [None, 224, 224, 3])
label_pb = tf.placeholder(tf.int32, [None])
logist, net = model(input_pb, is_train=False, pruning=proposal)

varis = tf.global_variables()
var_to_restore = [val for val in varis if '21' not in val.name and '22/kernel' not in val.name]
# var_to_restore = [val for val in varis if '53' not in val.name and 'fc3' not in val.name]
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
    # recover_var = ['conv_53/kernel', 'conv_53/bias', 'bn_53/beta', 'bn_53/gamma',
    #                'bn_53/moving_mean', 'bn_53/moving_variance', 'fc3_identity/W']
    recover_var = ['conv_21/kernel', 'conv_21/bias', 'bn_21/beta', 'bn_21/gamma',
                   'bn_21/moving_mean', 'bn_21/moving_variance',
                   'conv_22/kernel']
    for i in range(len(recover_var)):
        a = reader.get_tensor(recover_var[i])
        b = proposal['bn_21']
        c = [i for i in range(a.shape[-1]) if i not in b]
        if 'fc3_identity/W' in recover_var[i]:
            g = reader.get_tensor('conv_53/kernel')
            e = proposal['bn_53']
            f = [i for i in range(g.shape[-1]) if i not in e]
            d = a[f]
        elif '22' in recover_var[i]:
            c = [i for i in range(a.shape[-2]) if i not in b]
            d = a[..., c, :]
        else:
            d = a[..., c]
        sess.run(tf.assign(get_target_variable(recover_var[i]), d))

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
