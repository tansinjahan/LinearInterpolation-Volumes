from __future__ import division, print_function, absolute_import
import tensorflow.contrib.layers as lays

import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage import transform
from mpl_toolkits.mplot3d import Axes3D
from timeit import default_timer as timer

# --------------- Define parameters -----------------------------
batch_size = 10  # Number of samples in each batch
epoch_num = 20  # Number of epochs to train the network
lr = 0.001  # Learning rate
OUTPUT_SIZE = 32 # size of the output volume produced by decoder
INPUT_SIZE = 32 # size of the input volume given to the encoder
total_input = 50

def interpolationBetnLatentSpace(z1, z2):
    mx = 100
    mn = 0
    for t in range(mn, mx, 1):
        new_z = (1 - t) * z1 + t * z2

    return new_z

def resize_batch(imgs):
    # A function to resize a batch of MNIST images to (32, 32)
    # Args:
    #   imgs: a numpy array of size [batch_size, 32 X 32 x 32].
    # Returns:
    #   a numpy array of size [batch_size, 32, 32, 32].
    imgs = imgs.reshape((-1, 32, 32, 32, 1))

    resized_imgs = np.zeros((imgs.shape[0], 32, 32, 32, 1))
    for i in range(imgs.shape[0]):
        resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (32, 32, 32))
    print("This is resized image shape", resized_imgs.shape)
    return resized_imgs


def loadfile():
    input_file = np.array([])
    for i in range(21, 71):
        v = np.loadtxt('/home/gigl/Research/simple_autoencoder/Volume of shapes/MyTestFile' + str(i) + '.txt')
        image_matrix = np.reshape(v, (32, 32, 32)).astype(np.float32)
        ax_z, ax_x, ax_y = image_matrix.nonzero()
        input_fig = plt.figure()
        ax = input_fig.add_subplot(111, projection='3d')
        ax.scatter(ax_x, ax_y, ax_z, zdir='z', c='red')
        plt.savefig('input_data/demo' + str(i) + '.png')
        input_file = np.append(input_file, image_matrix)

    input_file = np.reshape(input_file, (50, 32 * 32 * 32))
    print("This is the shape of input for 50 shape", input_file.shape)
    return input_file


def autoencoder(inputs):
    # encoder
    # 32 x 32 x 32 x 1   -> 16 x 16 x 16 x 32
    # 16 x 16 x 16 x 32  ->  8 x 8 x 8 x 16
    # 8 x 8 x 8 x 16    ->  2 x 2 x 2 x 8
    net = lays.conv3d(inputs, 32, [5, 5, 5], stride=2, padding='SAME')
    net = lays.conv3d(net, 16, [5, 5, 5], stride=2, padding='SAME')
    net = lays.conv3d(net, 8, [5, 5, 5], stride=4, padding='SAME')
    #print("rank of Z", tf.rank(net))
    #net = lays.fully_connected(net, 1)
    latent_space = net
    # decoder
    # 2 x 2 x 2 x 8   ->  8 x 8 x 8 x 16
    # 8 x 8 x 8 x 16  ->  16 x 16 x 16 x 32
    # 16 x 16 x 16 x 32  ->  32 x 32 x 32 x 1
    net = lays.conv3d_transpose(net, 16, [5, 5, 5], stride=4, padding='SAME')
    net = lays.conv3d_transpose(net, 32, [5, 5, 5], stride=2, padding='SAME')
    net = lays.conv3d_transpose(net, 1, [5, 5, 5], stride=2, padding='SAME', activation_fn=tf.nn.tanh)
    return latent_space, net


def next_batch(next_batch_array, batchsize, offset):
    rowStart = offset * batchsize
    rowEnd = (rowStart + batchsize) - 1
    return next_batch_array[rowStart:rowEnd, :]


# --------------------read data set----------------------
input_file = loadfile()  # load 50 chairs as volume with shape [50,32768]

# ----------------- calculate the number of batches per epoch --------------------
batch_per_ep = input_file.shape[0] // batch_size  # batch per epoch will be 5 [input total = 50 divided by batch-size = 10 ]

ae_inputs = tf.placeholder(tf.float32, (None, 32, 32, 32, 1))  # input to the network (MNIST images)

l_space, ae_outputs = autoencoder(ae_inputs)  # create the Autoencoder network

# ----------------- calculate the loss and optimize the network--------------------------------
loss = tf.reduce_mean(tf.square(ae_outputs - ae_inputs))  # calculate the mean square error loss
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# -----------------------initialize the network---------------------------------
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    plot_loss = np.zeros([1, 2])
    for ep in range(epoch_num):  # epochs loop
        next_batch_array = input_file  # copy of input file to use for fetching next batch from input array
        for batch_n in range(batch_per_ep):  # batches loop
            batch_img = next_batch(next_batch_array, batch_size, batch_n)  # read a batch
            batch_img = resize_batch(batch_img)  # reshape the images to (32, 32)
            _, c = sess.run([train_op, loss], feed_dict={ae_inputs: batch_img})
            print('Epoch: {} - cost= {:.5f}'.format((ep + 1), c))
            plot_loss = np.append(plot_loss, [[(ep+1), c]], axis=0)

    # --------------------plot loss -------------------------------------
    print("This is plot_loss with shape", plot_loss, plot_loss.shape)
    #plt.plot(plot_loss[:, 0], plot_loss[:, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.plot(plot_loss[:, 0], plot_loss[:, 1], c='red')
    plt.savefig('output_data/loss.png')

    # ------------------test the trained network-------------------------------
    # batch_img = next_batch(input_file,1,0)

    # Test for the first input volume shape
    batch_img = input_file[0, :]
    batch_img = resize_batch(batch_img)
    recon_img = sess.run([l_space, ae_outputs], feed_dict={ae_inputs: batch_img})[1]
    l_space1 = sess.run([l_space, ae_outputs], feed_dict={ae_inputs: batch_img})[0]
    print("This is output shape of the decoder", recon_img.shape)
    out = np.reshape(recon_img, (OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_SIZE)).astype(np.float32)
    print("this is the shape of output volume", out.shape)

    # ----------------print the first line of the output shape ------------------------------

    print("this is type of the test shape after converting to numpy array", type(out), out.shape)
    for i in out[0, :, :]:
        print(i)

    # --------------------------plot the reconstructed images and their ground truths (inputs)----------------------

    # ------------------------Test method 1--------------------------------------------
    test_first_input = input_file[0, :]
    test_first_input = np.reshape(test_first_input, (OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_SIZE)).astype(np.float32)
    z, x, y = test_first_input.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, zdir='z', c='red')
    plt.savefig('output_data/demo_testMethod1.png')

    # -----------------------------test method 2----------------------------------------

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xdata = out[0, :, :]
    ydata = out[:, 1, :]
    zdata = out[:, :, 2]
    ax.scatter3D(xdata, ydata, zdata, zdir='z', c='red')
    plt.savefig('output_data/demo_testMethod2.png')

    # ------------------------------test method 3-----------------------------------------
    plotOutArr = np.array([])
    for i in range(0, OUTPUT_SIZE):
        for j in range(0, OUTPUT_SIZE):
            for k in range(0, OUTPUT_SIZE):
                if out[i, j, k] < 0.05:
                    plotOutArr = np.append(plotOutArr, 1)
                else:
                    plotOutArr = np.append(plotOutArr, 0)

    output_image = np.reshape(plotOutArr, (OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_SIZE)).astype(np.float32)
    z, x, y = output_image.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, zdir='z', c='red')
    plt.savefig('output_data/demo_testMethod3.png')

    # ------------------------------test method 4-----------------------------------------

    z, x, y = out.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, zdir='z', c='red')
    plt.savefig('output_data/demo_testMethod4.png')


    '''start = timer()
    new_z = interpolationBetnLatentSpace(l_space1)
    vectoradd_time = timer() - start
    print("Vector Add took %f seconds:", vectoradd_time)
    print("This is new  latent space", new_z)'''