import tensorflow as tf
import numpy as np
import time
from data import read_data, data_generator
from model import model
from varible import *

# reader = tf.train.NewCheckpointReader('/media/xinje/New Volume/hsq/ckpt/model_pruning/0/' + "ep000-step300-loss0.038")
# all_variables_shape = reader.get_variable_to_shape_map()
# a = reader.get_tensor("conv_52/kernel")
#
# reader2 = tf.train.NewCheckpointReader('./ckpt/' + "ep164-step45000-loss0.001")
# b =reader2.get_tensor("conv_52/kernel")


# def set_proposal():


layer_name = '51'
pruning_percentage = '0.9'
txts = ['./pruning_proposal/pruning_proposal_of_bn_{}_{}.txt'.format(layer_name, pruning_percentage),
        './pruning_proposal/pruning_proposal_of_bn_52_0.9.txt',
        './pruning_proposal/pruning_proposal_of_bn_53_0.9.txt']
proposal = dict()
for txt in txts:
    f = open(txt, 'r')
    proposal.update(eval(f.read()))
    f.close()


x_train, y_train, x_valid, y_valid, x_test, y_test = read_data('/home/xinje/hsq/data/dogVScat', 0.3, 0,
                                                               pos_path="/dog/", neg_path="/cat/")

input_pb = tf.placeholder(tf.float32, [None, 224, 224, 3])
label_pb = tf.placeholder(tf.int32, [None])
logist, net = model(input_pb, is_train=False, pruning=proposal)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    try:
        saver.restore(sess, '/media/xinje/New Volume/hsq/ckpt/model_pruning/14/' + "ep021-step11700-loss0.119")
        print("load ok!")
    except:
        print("ckpt文件不存在")
        raise

    data_yield = data_generator(x_valid, y_valid, is_train=False)
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
