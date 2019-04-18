from __future__ import division, print_function, absolute_import
import tensorflow.contrib.layers as lays
import tensorflow as tf
import keras_applications
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
from skimage import transform

z_dims = 20


def encoder(inputs):

    # encoder
    # 32 x 32 x 32 x 1   -> 16 x 16 x 16 x 32
    # 16 x 16 x 16 x 32  ->  8 x 8 x 8 x 16
    # 8 x 8 x 8 x 16    ->  2 x 2 x 2 x 8
    print("input type and shape", type(inputs), inputs.shape)
    net = lays.batch_norm(lays.conv3d(inputs, 32, [5, 5, 5], stride=2, padding='SAME', trainable=True),decay =0.9)
    net = lays.batch_norm(lays.conv3d(net, 16, [5, 5, 5], stride=2, padding='SAME', trainable=True),decay= 0.9)
    net = lays.batch_norm(lays.conv3d(net, 8, [5, 5, 5], stride=4, padding='SAME', trainable=True), decay=0.9)
    net = lays.batch_norm(lays.flatten(net),decay=0.9)

    z_mean = lays.fully_connected(net, z_dims)
    z_stdev = 0.5 * tf.nn.softplus(lays.fully_connected(net, z_dims))

    # Reparameterization trick for Variational Autoencoder
    samples = tf.random_normal([tf.shape(z_mean)[0], z_dims], mean=0, stddev=1, dtype=tf.float32)
    print("rank and shape of samples", tf.rank(samples))
    guessed_z = z_mean + tf.multiply(samples, tf.exp(z_stdev))
    print("rank and shape of guessed z", tf.rank(guessed_z))
    l_space = guessed_z
    return z_mean, z_stdev, l_space


def decoder(inputs):

    # decoder
    # 2 x 2 x 2 x 8   ->  8 x 8 x 8 x 16
    # 8 x 8 x 8 x 16  ->  16 x 16 x 16 x 32
    # 16 x 16 x 16 x 32  ->  32 x 32 x 32 x 1
    net = tf.layers.dense(inputs, 2 * 2 * 2 * 8, activation=tf.nn.relu)
    #net = tf.layers.dense(net, 2 * 2 * 2 * 8, activation=tf.nn.relu)
    net = tf.reshape(net, [-1, 2, 2, 2, 8])
    net = lays.batch_norm(lays.conv3d_transpose(net, 16, [5, 5, 5], stride=4, padding='SAME', trainable=True),decay=0.9)
    net = lays.batch_norm(lays.conv3d_transpose(net, 32, [5, 5, 5], stride=2, padding='SAME', trainable=True), decay=0.9)
    net = lays.conv3d_transpose(net, 1, [5, 5, 5], stride=2, padding='SAME', activation_fn=tf.nn.sigmoid, trainable=True)
    return net
