from __future__ import division, print_function, absolute_import
import tensorflow.contrib.layers as lays

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
from skimage import transform

def encoder(inputs):
    # encoder
    # 32 x 32 x 32 x 1   -> 16 x 16 x 16 x 32
    # 16 x 16 x 16 x 32  ->  8 x 8 x 8 x 16
    # 8 x 8 x 8 x 16    ->  2 x 2 x 2 x 8
    net = lays.conv3d(inputs, 32, [5, 5, 5], stride=2, padding='SAME', trainable=True)
    net = lays.conv3d(net, 16, [5, 5, 5], stride=2, padding='SAME', trainable=True)
    net = lays.conv3d(net, 8, [5, 5, 5], stride=4, padding='SAME', trainable=True)
    # print("rank of Z", tf.rank(net))
    # net = lays.fully_connected(net, 1)
    latent_space = net
    return latent_space

def decoder(inputs):
    # decoder
    # 2 x 2 x 2 x 8   ->  8 x 8 x 8 x 16
    # 8 x 8 x 8 x 16  ->  16 x 16 x 16 x 32
    # 16 x 16 x 16 x 32  ->  32 x 32 x 32 x 1
    net = lays.conv3d_transpose(inputs, 16, [5, 5, 5], stride=4, padding='SAME', trainable= True)
    net = lays.conv3d_transpose(net, 32, [5, 5, 5], stride=2, padding='SAME', trainable= True)
    net = lays.conv3d_transpose(net, 1, [5, 5, 5], stride=2, padding='SAME', activation_fn=tf.nn.tanh, trainable=True)
    return net