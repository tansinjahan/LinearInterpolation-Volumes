from __future__ import division, print_function, absolute_import
import tensorflow.contrib.layers as lays

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 10  # Number of samples in each batch
epoch_num = 5  # Number of epochs to train the network
lr = 0.001  # Learning rate


def resize_batch(imgs):
    # A function to resize a batch of MNIST images to (32, 32)
    # Args:
    #   imgs: a numpy array of size [batch_size, 28 X 28].
    # Returns:
    #   a numpy array of size [batch_size, 32, 32].
    #imgs = imgs.reshape((-1, 28, 28, 28, 1))

    resized_imgs = np.zeros((imgs.shape[0], 32, 32, 32, 1))
    for i in range(imgs.shape[0]):
        resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (32, 32, 32))
    print(resized_imgs.shape)
    return resized_imgs


def loadfile():
    input_file = np.array([])
    for i in range(21, 71):
        v = np.loadtxt('/home/gigl/Downloads/Volume of shapes/MyTestFile' + str(i) + '.txt')
        image_matrix = np.reshape(v, (32, 32, 32)).astype(np.float32)
        input_file = np.append(input_file, image_matrix)

    input_file = np.reshape(input_file, (50, 32 * 32 * 32))
    print("This is the shape of input for 50 shape", input_file.shape)
    return input_file


# read dataset
input_file = loadfile()  # load 50 chairs as volume with shape [50,32768]


def autoencoder(inputs):
    # encoder
    # 32 x 32 x 32 x 1   -> 16 x 16 x 16 x 32
    # 16 x 16 x 16 x 32  ->  8 x 8 x 8 x 16
    # 8 x 8 x 8 x 16    ->  2 x 2 x 2 x 8
    net = lays.conv3d(inputs, 32, [5, 5, 5], stride=2, padding='SAME')
    net = lays.conv3d(net, 16, [5, 5, 5], stride=2, padding='SAME')
    print(tf.shape(net))
    net = lays.conv3d(net, 8, [5, 5, 5], stride=4, padding='SAME')

    # latent_space = net
    # decoder
    # 2 x 2 x 2 x 8   ->  8 x 8 x 8 x 16
    # 8 x 8 x 8 x 16  ->  16 x 16 x 16 x 32
    # 16 x 16 x 16 x 32  ->  32 x 32 x 32 x 1
    net = lays.conv3d_transpose(net, 16, [5, 5, 5], stride=4, padding='SAME')
    net = lays.conv3d_transpose(net, 32, [5, 5, 5], stride=2, padding='SAME')
    net = lays.conv3d_transpose(net, 1, [5, 5, 5], stride=2, padding='SAME', activation_fn=tf.nn.tanh)
    return net


def next_batch(next_batch_array, batchsize, offset):
    rowStart = offset * batchsize
    rowEnd = (rowStart + batchsize) - 1
    return next_batch_array[rowStart:rowEnd, :]
    '''row,col = next_batch_array.shape # row should be 50 as our input file dim is [50,32768]
    if (next_batch_array != null):
        nextbatch = np.array([])
        for i in range (1, batchsize):
            n = input_file[i,:]
            nextbatch = np.append(nextbatch,n)
        next_batch_array = np.delete(next_batch_array,)'''


# calculate the number of batches per epoch


batch_per_ep = input_file.shape[0] // batch_size  # batch per epoch will be 5 [input total = 50 divided by batch-size = 10 ]

ae_inputs = tf.placeholder(tf.float32, (None, 32, 32, 32, 1))  # input to the network (MNIST images)

ae_outputs = autoencoder(ae_inputs)  # create the Autoencoder network

# calculate the loss and optimize the network
loss = tf.reduce_mean(tf.square(ae_outputs - ae_inputs))  # calculate the mean square error loss
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# initialize the network
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for ep in range(epoch_num):  # epochs loop
        next_batch_array = input_file  # copy of input file to use for fetching next batch from input array
        for batch_n in range(batch_per_ep):  # batches loop
            batch_img = next_batch(next_batch_array, batch_size, batch_n)  # read a batch
            #batch_img = batch_img.reshape((-1, 28, 28, 28, 1))  # reshape each sample to an (28, 28) image
            batch_img = resize_batch(batch_img)  # reshape the images to (32, 32)
            _, c = sess.run([train_op, loss], feed_dict={ae_inputs: batch_img})
            print('Epoch: {} - cost= {:.5f}'.format((ep + 1), c))


 # test the trained network
    batch_img = next_batch(input_file,10,0)
    batch_img = resize_batch(batch_img)
    recon_img = sess.run([ae_outputs], feed_dict={ae_inputs: batch_img})[0]

    # plot the reconstructed images and their ground truths (inputs)
    plt.figure(1)
    plt.title('Reconstructed Images')
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(recon_img[i, ..., 0], cmap='gray')
    plt.figure(2)
    plt.title('Input Images')
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(batch_img[i, ..., 0], cmap='gray')
    plt.show()