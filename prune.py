import tensorflow as tf
import numpy as np
import time
from data import read_data, data_generator
from model import model
from varible import *

from train import *
from matplotlib import pyplot as plt


def get_target_variable(name):
    all = tf.global_variables()
    for i in range(len(all)):
        if name in all[i].name:
            return all[i]
    return None


layer_name = '53'
pruning_percentage = '0.1'
f = open('./pruning_proposal/pruning_proposal_of_bn_{}_{}.txt'.format(layer_name, pruning_percentage), 'r')
proposal = eval(f.read())
f.close()

x_train, y_train, x_valid, y_valid, x_test, y_test = read_data('/home/xinje/hsq/data/dogVScat', 0.3, 0,
                                                               pos_path="/dog/", neg_path="/cat/")

batch_size = Gb_batch_size
learning_rate = Gb_learning_rate
log_dir = Gb_ckpt_dir
final_dir = Gb_ckpt_dir
n_epoch = Gb_epoch
n_step_epoch = int(len(y_train) / batch_size)
save_frequency = Gb_save_frequency

input_pb = tf.placeholder(tf.float32, [None, 224, 224, 3])
label_pb = tf.placeholder(tf.int32, [None])
logist, net = model(input_pb, is_train=True, pruning=proposal)
loss_op = losses(logits=logist, labels=label_pb)
trainable_var = [var for var in tf.trainable_variables() if layer_name in var.name or 'fc3_identity' in var.name]
train_op = trainning(loss_op, learning_rate=learning_rate, var_list=trainable_var)

# saver = tf.train.Saver(max_to_keep=100)
summary_op = tf.summary.merge_all()
temp = ''

varis = tf.global_variables()
var_to_restore = [val for val in varis if
                  layer_name not in val.name and '{}/kernel'.format(int(layer_name) + 1) not in val.name]
if layer_name == '53':
    var_to_restore = [val for val in varis if '53' not in val.name and 'fc3_identity/W' not in val.name]
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
    recover_var = ['conv_{}/kernel'.format(layer_name), 'conv_{}/bias'.format(layer_name),
                   'bn_{}/beta'.format(layer_name), 'bn_{}/gamma'.format(layer_name),
                   'bn_{}/moving_mean'.format(layer_name), 'bn_{}/moving_variance'.format(layer_name),
                   'fc3_identity/W' if layer_name == '53' else 'conv_{}/kernel'.format(int(layer_name) + 1)]
    for i in range(len(recover_var)):
        a = reader.get_tensor(recover_var[i])
        b = proposal['bn_{}'.format(layer_name)]
        c = [i for i in range(a.shape[-1]) if i not in b]
        if 'fc3_identity/W' in recover_var[i]:
            g = reader.get_tensor('conv_53/kernel')
            e = proposal['bn_53']
            f = [i for i in range(g.shape[-1]) if i not in e]
            d = a[f]
        elif str(int(layer_name) + 1) in recover_var[i]:
            c = [i for i in range(a.shape[-2]) if i not in b]
            d = a[..., c, :]
        else:
            d = a[..., c]
        sess.run(tf.assign(get_target_variable(recover_var[i]), d))

    # a = tf.trainable_variables()
    # trainable_collection = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
    # a = tf.trainable_variables()
    saver = tf.train.Saver(max_to_keep=100)
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
