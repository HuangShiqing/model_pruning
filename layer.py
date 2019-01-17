from tensorlayer.layers import *
import tensorflow as tf


class AddLayer(Layer):
    def __init__(self, prev_layer, name='add_layer'):
        super(AddLayer, self).__init__(prev_layer=prev_layer, name=name)

        # logging.info("ConcatLayer %s: axis: %d" % (self.name, concat_dim))

        self.outputs = tf.add(self.inputs[0], self.inputs[1], name=name)
        self._add_layers(self.outputs)


def Conv_Bn(net, n_filter, filter_size, strides, act, bn, is_train, padding, name):
    # TODO:添加w正则化
    if bn is True:
        net = Conv2d(net, n_filter=n_filter, filter_size=(filter_size, filter_size), strides=(strides, strides),
                     act=tf.identity, padding=padding, b_init=None, name=name + '_conv')
        net = BatchNormLayer(net, epsilon=1e-6, act=act, is_train=is_train, name=name + '_bn')
    else:
        net = Conv2d(net, n_filter=n_filter, filter_size=(filter_size, filter_size), strides=(strides, strides),
                     act=act, padding=padding, name=name + '_conv')
    return net


def ShortCut(net, res_net, name):
    res_net_channel = res_net.outputs.shape[-1].value
    res_net_w = res_net.outputs.shape[-2].value

    net_channel = net.outputs.shape[-1].value
    net_w = net.outputs.shape[-2].value

    strides = int(round(res_net_w / net_w))
    if res_net_channel != net_channel or strides > 1:
        res_net = Conv2d(res_net, n_filter=net.outputs.shape[-1], filter_size=(1, 1), strides=(strides, strides),
                         act=tf.identity, padding='VALID', name=name + '_shortcut_conv')
    net = AddLayer([net, res_net], name=name + '_shortcut_add')
    return net
