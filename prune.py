import numpy as np
import tensorflow as tf
import time
import os
from sklearn.utils import shuffle
import shutil

from varible import *
from model import *
from proposal_L1 import *
from proposal_APoZ import *
from train import optimizer_sgd, losses
from data import read_data, data_generator


def random_proposal(n_filter):
    name = ['11', '12', '21', '22', '31', '32', '33', '41', '42', '43', '51', '52', '53']
    n_all = [n_filter[f] for f in n_filter]
    all = 0
    for n in n_all:
        all += n
    rand = np.random.randint(all)
    for i in name:
        if rand < n_filter[i]:
            proposal = {i: [rand]}
            return proposal
        else:
            rand -= n_filter[i]


def get_target_variable(name):
    all = tf.global_variables()
    for i in range(len(all)):
        if name in all[i].name:
            return all[i]
    return None


def restore_part_var(load_ckpt_dir, load_ckpt_name, proposal, layer_names, layer_index, sess):
    # 创建一个读写器
    reader = tf.train.NewCheckpointReader(os.path.join(load_ckpt_dir, load_ckpt_name))
    var_part_restore = ['{}_conv/kernel'.format(layer_names[layer_index]),
                        # '{}_conv/bias'.format(layer_names[layer_index]),
                        '{}_bn/beta'.format(layer_names[layer_index]),
                        '{}_bn/gamma'.format(layer_names[layer_index]),
                        '{}_bn/moving_mean'.format(layer_names[layer_index]),
                        '{}_bn/moving_variance'.format(layer_names[layer_index]),
                        'dense/W' if layer_names[layer_index] == '53' else '{}_conv/kernel'.format(
                            int(layer_names[layer_index + 1]))]
    for i in range(len(var_part_restore)):
        a = reader.get_tensor(var_part_restore[i])
        b = proposal['{}'.format(layer_names[layer_index])]
        c = [i for i in range(a.shape[-1]) if i not in b]

        # 被裁剪的那一层参数恢复
        if str(int(layer_names[layer_index])) in var_part_restore[i]:
            d = a[..., c]
        else:
            # 如果被裁减的是最后一层的话，需要恢复'fc3_identity/W'
            if layer_index == 12:
                g = reader.get_tensor('53_conv/kernel')
                e = proposal['53']
                f = [i for i in range(g.shape[-1]) if i not in e]
                d = a[f]
            # 恢复被裁剪的后一层参数
            else:
                # elif str(int(layer_names[layer_index + 1])) in var_part_restore[i]:
                c = [i for i in range(a.shape[-2]) if i not in b]
                d = a[..., c, :]
        sess.run(tf.assign(get_target_variable(var_part_restore[i]), d))


n_filter = {'11': 64, '12': 64, '21': 128, '22': 128, '31': 256, '32': 256, '33': 256, '41': 512, '42': 512, '43': 512,
            '51': 512, '52': 512, '53': 512}
x_train, y_train, x_valid, y_valid = read_data(Gb_data_dir)
fine_turning_step = Gb_fine_turning_step
batch_size = Gb_batch_size
learning_rate = Gb_learning_rate
n_step_epoch = int(len(y_train) / batch_size)

last_ckpt = ''
for channel in range(0, 4224):
    tf.reset_default_graph()
    load_ckpt_dir = './ckpt/' if last_ckpt == '' else os.path.join(Gb_ckpt_dir, str(channel - 1))
    load_ckpt_name = 'ep006-step3000-loss5.232' if last_ckpt == '' else last_ckpt
    final_dir = os.path.join(Gb_ckpt_dir, str(channel))
    log_dir = final_dir

    # proposal = APoZ_proposal(n_filter, load_ckpt_dir, load_ckpt_name)
    proposal = L1_proposal(n_filter, load_ckpt_dir, load_ckpt_name)
    # proposal = random_proposal(n_filter)
    # proposal = {'11': [20]}
    tf.reset_default_graph()
    layer_names = ['11', '12', '21', '22', '31', '32', '33', '41', '42', '43', '51', '52', '53']
    layer_index = layer_names.index(list(proposal.keys())[0])
    n_filter[list(proposal.keys())[0]] -= 1

    input_pb = tf.placeholder(tf.float32, [None, 224, 224, 3])
    label_pb = tf.placeholder(tf.int32, [None])
    logist, net = vgg16_adjusted(input_pb, n_filter=n_filter)
    loss_op = losses(logits=logist, labels=label_pb)
    train_op = optimizer_sgd(loss_op, learning_rate=learning_rate)

    varis = tf.global_variables()
    if layer_names[layer_index] == '53':
        var_all_restore = [val for val in varis if '53' not in val.name and 'dense/W' not in val.name]
    else:
        var_all_restore = [val for val in varis if layer_names[layer_index] not in val.name and '{}_conv/kernel'.format(
            int(layer_names[layer_index + 1])) not in val.name]
    restore_saver = tf.train.Saver(var_all_restore)

    saver = tf.train.Saver(max_to_keep=1000)
    summary_op = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        try:
            restore_saver.restore(sess, os.path.join(load_ckpt_dir, load_ckpt_name))
            print("load ok!")
        except:
            print("ckpt文件不存在")
            raise

        restore_part_var(load_ckpt_dir, load_ckpt_name, proposal, layer_names, layer_index, sess)

        train_writer = tf.summary.FileWriter(log_dir, sess.graph)
        step = 0
        epoch = 0
        step_epoch = 0
        x_train, y_train = shuffle(x_train, y_train)
        data_yield = data_generator(x_train, y_train)
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

            if step == fine_turning_step:
                break
        print("Save model " + "!" * 10)
        save_path = saver.save(sess,
                               os.path.join(final_dir, 'ep{0:03d}-step{1:d}-loss{2:.3f}'.format(epoch, step, loss)))
        last_ckpt = 'ep{0:03d}-step{1:d}-loss{2:.3f}'.format(epoch, step, loss)

        train_writer.close()
        if channel % 50 == 0 and channel > 0:
            for i in range(channel - 49, channel):
                # os.rmdir(os.path.join(Gb_ckpt_dir, str(i)))
                # TODO:删除一直失败
                shutil.rmtree(os.path.join(Gb_ckpt_dir, str(i)), ignore_errors=True)

    if channel % 50 == 0:
        tf.reset_default_graph()
        data_yield = data_generator(x_valid, y_valid, is_train=False)
        input_pb = tf.placeholder(tf.float32, [None, 224, 224, 3])
        logist, net = vgg16_adjusted(input_pb, n_filter=n_filter, is_train=False)
        saver = tf.train.Saver(max_to_keep=1000)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            try:
                saver.restore(sess, os.path.join(final_dir, last_ckpt))
                print("load ok!")
            except:
                print("ckpt文件不存在")
                raise

            error_num = 0
            i = 0
            for img, lable in data_yield:
                logist_out = sess.run(logist, feed_dict={input_pb: img})
                logist_out = np.argmax(logist_out, axis=-1)
                a = np.equal(logist_out, list(map(int, lable)))
                a = list(a)
                error_num += a.count(False)
                i += 1
            print('error: ', str(error_num), ' in ', str(i * Gb_batch_size))

            with open(os.path.join(Gb_ckpt_dir, 'log.txt'), 'a') as f:
                f.write(str(channel) + str(proposal) + '\n')
                f.write('error: ' + str(error_num) + ' in ' + str(i * Gb_batch_size) + '\n')
    else:
        with open(os.path.join(Gb_ckpt_dir, 'log.txt'), 'a') as f:
            f.write(str(channel) + str(proposal) + '\n')
