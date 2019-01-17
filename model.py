from layer import *


def vgg16_adjusted(x, n_filter, is_train=True):
    n = 200

    # n_filter = {'11': 64, '12': 64, '21': 128, '22': 128, '31': 256, '32': 256,
    #             '33': 256, '41': 512, '42': 512, '43': 512,
    #             '51': 512, '52': 512, '53': 512}

    net = InputLayer(x, name='input')
    # conv1
    net = Conv_Bn(net, n_filter=n_filter['11'], filter_size=3, strides=1, act=tf.nn.relu, bn=True, is_train=is_train,
                  padding='SAME', name='11')
    net = Conv_Bn(net, n_filter=n_filter['12'], filter_size=3, strides=1, act=tf.nn.relu, bn=True, is_train=is_train,
                  padding='SAME', name='12')
    net = MaxPool2d(net, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')
    # conv2
    net = Conv_Bn(net, n_filter=n_filter['21'], filter_size=3, strides=1, act=tf.nn.relu, bn=True, is_train=is_train,
                  padding='SAME', name='21')
    net = Conv_Bn(net, n_filter=n_filter['22'], filter_size=3, strides=1, act=tf.nn.relu, bn=True, is_train=is_train,
                  padding='SAME', name='22')
    net = MaxPool2d(net, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')
    # conv3
    net = Conv_Bn(net, n_filter=n_filter['31'], filter_size=3, strides=1, act=tf.nn.relu, bn=True, is_train=is_train,
                  padding='SAME', name='31')
    net = Conv_Bn(net, n_filter=n_filter['32'], filter_size=3, strides=1, act=tf.nn.relu, bn=True, is_train=is_train,
                  padding='SAME', name='32')
    net = Conv_Bn(net, n_filter=n_filter['33'], filter_size=3, strides=1, act=tf.nn.relu, bn=True, is_train=is_train,
                  padding='SAME', name='33')
    net = MaxPool2d(net, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')
    # conv4
    net = Conv_Bn(net, n_filter=n_filter['41'], filter_size=3, strides=1, act=tf.nn.relu, bn=True, is_train=is_train,
                  padding='SAME', name='41')
    net = Conv_Bn(net, n_filter=n_filter['42'], filter_size=3, strides=1, act=tf.nn.relu, bn=True, is_train=is_train,
                  padding='SAME', name='42')
    net = Conv_Bn(net, n_filter=n_filter['43'], filter_size=3, strides=1, act=tf.nn.relu, bn=True, is_train=is_train,
                  padding='SAME', name='43')
    net = MaxPool2d(net, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool4')
    # conv5
    net = Conv_Bn(net, n_filter=n_filter['51'], filter_size=3, strides=1, act=tf.nn.relu, bn=True, is_train=is_train,
                  padding='SAME', name='51')
    net = Conv_Bn(net, n_filter=n_filter['52'], filter_size=3, strides=1, act=tf.nn.relu, bn=True, is_train=is_train,
                  padding='SAME', name='52')
    net = Conv_Bn(net, n_filter=n_filter['53'], filter_size=3, strides=1, act=tf.nn.relu, bn=True, is_train=is_train,
                  padding='SAME', name='53')

    net = GlobalMeanPool2d(net)
    net = DenseLayer(net, n_units=n, act=None, name='dense')
    return net.outputs, net


def small_model(x, is_train=True):
    act = tf.nn.relu
    n = 2

    network = InputLayer(x, name='input')
    network = Conv2dLayer(network, act=act, shape=(5, 5, 3, 32), strides=(1, 1, 1, 1), padding='SAME', name='conv_1')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool_1')
    network = Conv2dLayer(network, act=act, shape=(5, 5, 32, 64), strides=(1, 1, 1, 1), padding='SAME', name='conv_2')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool_2')
    network = Conv2dLayer(network, act=act, shape=(5, 5, 64, 128), strides=(1, 1, 1, 1), padding='SAME', name='conv_3')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool_3')
    network = Conv2dLayer(network, act=act, shape=(5, 5, 128, 256), strides=(1, 1, 1, 1), padding='SAME', name='conv_4')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool_4')

    network = FlattenLayer(network, name='flatten_layer')
    # TODO:maybe something wrong with dropout's fix
    network = DropoutLayer(network, keep=0.5, is_fix=is_train, name='drop_1')
    network = DenseLayer(network, n_units=256, act=tf.nn.relu, name='dense_1')
    network = DropoutLayer(network, keep=0.5, is_fix=is_train, name='drop_2')
    network = DenseLayer(network, n_units=n, act=tf.identity, name='dense_2')
    y = network.outputs
    return y, network


def resnet_101(x, is_train=True):
    n = 2
    net = InputLayer(x, name='input')
    net = Conv_Bn(net, n_filter=64, filter_size=7, strides=2, act=tf.nn.relu, bn=True, padding='SAME',
                  is_train=is_train, name='1')
    net1 = MaxPool2d(net, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool_1')

    # first block ,repetitions 3
    start_layer = 2
    for i in range(3):
        if i != 0:
            net = BatchNormLayer(net1, epsilon=1e-6, act=tf.nn.relu, is_train=is_train,
                                 name=str(i * 3 + start_layer - 1) + '_shortcut_bn')
        net = Conv_Bn(net if i != 0 else net1, n_filter=64, filter_size=1, strides=1, act=tf.nn.relu, bn=True,
                      padding='SAME', is_train=is_train, name=str(i * 3 + start_layer))
        net = Conv_Bn(net, n_filter=64, filter_size=3, strides=1, act=tf.nn.relu, bn=True, padding='SAME',
                      is_train=is_train, name=str(i * 3 + start_layer + 1))
        net = Conv_Bn(net, n_filter=256, filter_size=1, strides=1, act=tf.identity, bn=False, padding='SAME',
                      is_train=is_train, name=str(i * 3 + start_layer + 2))
        net1 = ShortCut(net, net1, name=str(i * 3 + start_layer + 2))

    # second block ,repetitions 4
    start_layer = 11
    for i in range(4):
        net = BatchNormLayer(net1, epsilon=1e-6, act=tf.nn.relu, is_train=is_train,
                             name=str(i * 3 + start_layer - 1) + '_shortcut_bn')
        net = Conv_Bn(net, n_filter=128, filter_size=1, strides=2 if i == 0 else 1, act=tf.nn.relu, bn=True,
                      padding='SAME', is_train=is_train, name=str(i * 3 + start_layer))
        net = Conv_Bn(net, n_filter=128, filter_size=3, strides=1, act=tf.nn.relu, bn=True, padding='SAME',
                      is_train=is_train, name=str(i * 3 + start_layer + 1))
        net = Conv_Bn(net, n_filter=512, filter_size=1, strides=1, act=tf.identity, bn=False, padding='SAME',
                      is_train=is_train, name=str(i * 3 + start_layer + 2))
        net1 = ShortCut(net, net1, name=str(i * 3 + start_layer + 2))

    # third block ,repetitions 23
    start_layer = 23
    for i in range(23):
        net = BatchNormLayer(net1, epsilon=1e-6, act=tf.nn.relu, is_train=is_train,
                             name=str(i * 3 + start_layer - 1) + '_shortcut_bn')
        net = Conv_Bn(net, n_filter=256, filter_size=1, strides=2 if i == 0 else 1, act=tf.nn.relu, bn=True,
                      padding='SAME', is_train=is_train, name=str(i * 3 + start_layer))
        net = Conv_Bn(net, n_filter=256, filter_size=3, strides=1, act=tf.nn.relu, bn=True, padding='SAME',
                      is_train=is_train, name=str(i * 3 + start_layer + 1))
        net = Conv_Bn(net, n_filter=1024, filter_size=1, strides=1, act=tf.identity, bn=False, padding='SAME',
                      is_train=is_train, name=str(i * 3 + start_layer + 2))
        net1 = ShortCut(net, net1, name=str(i * 3 + start_layer + 2))

    # forth block ,repetitions 3
    start_layer = 92
    for i in range(3):
        net = BatchNormLayer(net1, epsilon=1e-6, act=tf.nn.relu, is_train=is_train,
                             name=str(i * 3 + start_layer - 1) + '_shortcut_bn')
        net = Conv_Bn(net, n_filter=512, filter_size=1, strides=2, act=tf.nn.relu, bn=True, padding='SAME',
                      is_train=is_train, name=str(i * 3 + start_layer))
        net = Conv_Bn(net, n_filter=512, filter_size=3, strides=1, act=tf.nn.relu, bn=True, padding='SAME',
                      is_train=is_train, name=str(i * 3 + start_layer + 1))
        net = Conv_Bn(net, n_filter=2048, filter_size=1, strides=1, act=tf.identity, bn=False, padding='SAME',
                      is_train=is_train, name=str(i * 3 + start_layer + 2))
        net1 = ShortCut(net, net1, name=str(i * 3 + start_layer + 2))

    net = BatchNormLayer(net1, epsilon=1e-6, act=tf.nn.relu, is_train=is_train, name='100_shortcut_bn')
    net = GlobalMeanPool2d(net)
    net = DenseLayer(net, n_units=n, act=None, name='101_dense')
    return net.outputs, net


if __name__ == '__main__':
    input_pb = tf.placeholder(tf.float32, [None, 224, 224, 3])
    label_pb = tf.placeholder(tf.int32, [None])
    logist, net = resnet_101(input_pb)
    exit()
