import tensorflow as tf
import numpy as np
import time
from data import read_data, data_generator
from model import model
from varible import *

from matplotlib import pyplot as plt

f = open('./pruning_proposal/pruning_proposal_of_bn_53.txt', 'r')
proposal = eval(f.read())
f.close()

x_train, y_train, x_valid, y_valid, x_test, y_test = read_data('D:/DeepLearning/data2/dogVScat/train', 0.3, 0,
                                                               pos_path="/dog/", neg_path="/cat/")
input_pb = tf.placeholder(tf.float32, [None, 224, 224, 3])
label_pb = tf.placeholder(tf.int32, [None])
logist, net = model(input_pb, is_train=False, pruning={'conv_53': len(proposal)})

varis = tf.global_variables()
var_to_restore = [val for val in varis if 'Adam' ]
saver = tf.train.Saver(var_to_restore)
saver = tf.train.Saver()
with tf.Session() as sess:
    try:
        saver.restore(sess, './ckpt/' + "ep164-step45000-loss0.001")
        print("load ok!")
    except:
        print("ckpt文件不存在")
        raise
