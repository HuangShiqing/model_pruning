from tensorlayer.layers import *
import tensorflow as tf
import tensorlayer as tl

from tensorlayer.layers.pooling import *


def model(x, is_train=True):
    act = tf.nn.relu
    n = 2

    net_in = InputLayer(x, name='input')
    # conv1
    net = Conv2d(net_in, 64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_1')
    net = BatchNormLayer(net, epsilon=1e-3, act=act, is_train=is_train, name='bn_11')
    net = Conv2d(net, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_2')
    net = BatchNormLayer(net, epsilon=1e-3, act=act, is_train=is_train, name='bn_12')
    net = MaxPool2d(net, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')

    # conv2
    net = Conv2d(net, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_1')
    net = BatchNormLayer(net, epsilon=1e-3, act=act, is_train=is_train, name='bn_21')
    net = Conv2d(net, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_2')
    net = BatchNormLayer(net, epsilon=1e-3, act=act, is_train=is_train, name='bn_22')
    net = MaxPool2d(net, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')

    # conv3
    net = Conv2d(net, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_1')
    net = BatchNormLayer(net, epsilon=1e-3, act=act, is_train=is_train, name='bn_31')
    net = Conv2d(net, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_2')
    net = BatchNormLayer(net, epsilon=1e-3, act=act, is_train=is_train, name='bn_32')
    net = Conv2d(net, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_3')
    net = BatchNormLayer(net, epsilon=1e-3, act=act, is_train=is_train, name='bn_33')
    net = MaxPool2d(net, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')

    # conv4
    net = Conv2d(net, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_1')
    net = BatchNormLayer(net, epsilon=1e-3, act=act, is_train=is_train, name='bn_41')
    net = Conv2d(net, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_2')
    net = BatchNormLayer(net, epsilon=1e-3, act=act, is_train=is_train, name='bn_42')
    net = Conv2d(net, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_3')
    net = BatchNormLayer(net, epsilon=1e-3, act=act, is_train=is_train, name='bn_43')
    net = MaxPool2d(net, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool4')

    # conv5
    net = Conv2d(net, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_1')
    net = BatchNormLayer(net, epsilon=1e-3, act=act, is_train=is_train, name='bn_51')
    net = Conv2d(net, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_2')
    net = BatchNormLayer(net, epsilon=1e-3, act=act, is_train=is_train, name='bn_52')
    net = Conv2d(net, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_3')
    net = BatchNormLayer(net, epsilon=1e-3, act=act, is_train=is_train, name='bn_53')
    # net = MaxPool2d(net, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool5')
    #
    # net = FlattenLayer(net, name='flatten')
    # net = DenseLayer(net, n_units=4096, act=tf.nn.relu, name='fc1_relu')
    # net = DenseLayer(net, n_units=4096, act=tf.nn.relu, name='fc2_relu')
    # net = DenseLayer(net, n_units=n, act=None, name='fc3_relu')
    net = GlobalMeanPool2d(net)
    net = DenseLayer(net, n_units=n, act=None, name='fc3_identity')

    return net.outputs, net


if __name__ == '__main__':
    input_pb = tf.placeholder(tf.float32, [None, 224, 224, 3])
    label_pb = tf.placeholder(tf.int32, [None])
    logist = model(input_pb)
    exit()
