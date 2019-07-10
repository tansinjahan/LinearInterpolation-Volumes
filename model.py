from __future__ import division, print_function, absolute_import
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
    net = lays.conv3d(inputs, 32, [4, 4, 4], stride=2, padding='SAME', trainable=True) #[16,16,16,32]
    net = lays.batch_norm(net, decay=0.999)
    net = lays.conv3d(net, 16, [4, 4, 4], stride=2, padding='SAME', trainable=True) #[8,8,8,16]
    net = lays.batch_norm(net, decay=0.999)
    net = lays.conv3d(net, 8, [4, 4, 4], stride=2, padding='SAME', trainable=True) #[4,4,4,8]
    net = lays.batch_norm(net, decay=0.999)
    #net = lays.fully_connected(net, 100) # fc = [100]

    #net = lays.flatten(net,100)
    print("this is net", net)
    print("this is shape", tf.shape(net))

    return net

def decoder(inputs):
    # decoder
    # 2 x 2 x 2 x 8   ->  8 x 8 x 8 x 16
    # 8 x 8 x 8 x 16  ->  16 x 16 x 16 x 32
    # 16 x 16 x 16 x 32  ->  32 x 32 x 32 x 1
    #net = lays.fully_connected(inputs, 512)
    #net = tf.reshape(net, [4,4,4,8])
    #net = lays.conv3d_transpose(net, 8, [4, 4, 4], stride=2, padding='SAME', trainable=True)
    net = lays.conv3d_transpose(inputs, 16, [4, 4, 4], stride=2, padding='SAME', trainable= True)
    net = lays.batch_norm(net, decay=0.999)
    net = lays.conv3d_transpose(net, 32, [4, 4, 4], stride=2, padding='SAME', trainable= True)
    net = lays.batch_norm(net, decay=0.999)
    net = lays.conv3d_transpose(net, 1, [4, 4, 4], stride=2, padding='SAME', activation_fn=tf.nn.sigmoid, trainable=True)
    return net