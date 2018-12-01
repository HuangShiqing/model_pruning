import tensorflow as tf
import numpy as np
import time
import os
from matplotlib import pyplot as plt

from data import read_data, data_generator
from model import model
from varible import *
from train import trainning, losses


def get_target_variable(name):
    all = tf.global_variables()
    for i in range(len(all)):
        if name in all[i].name:
            return all[i]
    return None


continue_flag = True
load_ckpt_dir = '/media/xinje/New Volume/hsq/ckpt/model_pruning/13/'
load_ckpt_name = "ep043-step23700-loss0.098"
final_dir = '/media/xinje/New Volume/hsq/ckpt/model_pruning/14/'
log_dir = final_dir

layer_name = '51'
pruning_percentage = '0.9'
txts = ['./pruning_proposal/pruning_proposal_of_bn_{}_{}.txt'.format(layer_name, pruning_percentage),
        './pruning_proposal/pruning_proposal_of_bn_52_0.9.txt',
        './pruning_proposal/pruning_proposal_of_bn_53_0.9.txt'
        ]
proposal = dict()
for txt in txts:
    f = open(txt, 'r')
    proposal.update(eval(f.read()))
    f.close()

x_train, y_train, x_valid, y_valid, x_test, y_test = read_data('/home/xinje/hsq/data/dogVScat', 0.3, 0,
                                                               pos_path="/dog/", neg_path="/cat/")

batch_size = Gb_batch_size
learning_rate = Gb_learning_rate
# log_dir = Gb_ckpt_dir
# final_dir = Gb_ckpt_dir
n_epoch = Gb_epoch
n_step_epoch = int(len(y_train) / batch_size)
save_frequency = Gb_save_frequency

input_pb = tf.placeholder(tf.float32, [None, 224, 224, 3])
label_pb = tf.placeholder(tf.int32, [None])
logist, net = model(input_pb, is_train=True, pruning=proposal)
loss_op = losses(logits=logist, labels=label_pb)

var_trainable = [var for var in tf.trainable_variables() if layer_name in var.name]
# if continue_flag is True:
#     var_trainable = None
train_op = trainning(loss_op, learning_rate=learning_rate, var_list=var_trainable)

summary_op = tf.summary.merge_all()
temp = ''

varis = tf.global_variables()
var_all_restore = [val for val in varis if
                   layer_name not in val.name and '{}/kernel'.format(int(layer_name) + 1) not in val.name]
if layer_name == '53':
    var_all_restore = [val for val in varis if '53' not in val.name and 'fc3_identity/W' not in val.name]
if continue_flag is True:
    var_all_restore = None
saver = tf.train.Saver(var_all_restore)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    try:
        saver.restore(sess, load_ckpt_dir + load_ckpt_name)
        print("load ok!")
    except:
        print("ckpt文件不存在")
        raise

    if continue_flag is False:
        # 创建一个读写器
        reader = tf.train.NewCheckpointReader(load_ckpt_dir + load_ckpt_name)
        var_part_restore = ['conv_{}/kernel'.format(layer_name), 'conv_{}/bias'.format(layer_name),
                            'bn_{}/beta'.format(layer_name), 'bn_{}/gamma'.format(layer_name),
                            'bn_{}/moving_mean'.format(layer_name), 'bn_{}/moving_variance'.format(layer_name),
                            'fc3_identity/W' if layer_name == '53' else 'conv_{}/kernel'.format(int(layer_name) + 1)]
        for i in range(len(var_part_restore)):
            a = reader.get_tensor(var_part_restore[i])
            b = proposal['bn_{}'.format(layer_name)]
            c = [i for i in range(a.shape[-1]) if i not in b]
            if 'fc3_identity/W' in var_part_restore[i]:
                g = reader.get_tensor('conv_53/kernel')
                e = proposal['bn_53']
                f = [i for i in range(g.shape[-1]) if i not in e]
                d = a[f]
            elif str(int(layer_name) + 1) in var_part_restore[i]:
                c = [i for i in range(a.shape[-2]) if i not in b]
                d = a[..., c, :]
            else:
                d = a[..., c]
            sess.run(tf.assign(get_target_variable(var_part_restore[i]), d))

    saver = tf.train.Saver(max_to_keep=1000)
    train_writer = tf.summary.FileWriter(log_dir, sess.graph)
    step = 0
    min_loss = 10000000
    for epoch in range(n_epoch):
        step_epoch = 0
        # TODO shuffle chunks
        data_yield = data_generator(x_train, y_train, is_train=True)

        for img, lable in data_yield:
            step += 1
            step_epoch += 1
            start_time = time.time()
            loss, _, summary_str = sess.run([loss_op, train_op, summary_op],
                                            feed_dict={input_pb: img, label_pb: lable})
            train_writer.add_summary(summary_str, step)
            # 每step打印一次该step的loss
            print("Loss %fs  : Epoch %d  %d/%d: Step %d  took %fs" % (
                loss, epoch, step_epoch, n_step_epoch, step, time.time() - start_time))

            if step % save_frequency == 0:
                print("Save model " + "!" * 10)
                save_path = saver.save(sess,
                                       final_dir + 'ep{0:03d}-step{1:d}-loss{2:.3f}'.format(epoch, step, loss))
                if loss < min_loss:
                    min_loss = loss
                else:
                    try:
                        os.remove(final_dir + temp + '.data-00000-of-00001')
                        os.remove(final_dir + temp + '.index')
                        os.remove(final_dir + temp + '.meta')
                    except:
                        pass
                    temp = 'ep{0:03d}-step{1:d}-loss{2:.3f}'.format(epoch, step, loss)
    exit()
