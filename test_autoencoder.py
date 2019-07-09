from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from timeit import default_timer as timer
from mpl_toolkits.mplot3d import Axes3D
import meshlab_visualize
import model as md
import for_plot

#import theano.tensor as T

# --------------- Define parameters -----------------------------
batch_size = 10  # Number of samples in each batch
epoch_num = 10  # Number of epochs to train the network
lr = 0.001  # Learning rate
OUTPUT_SIZE = 32 # size of the output volume produced by decoder
INPUT_SIZE = 32 # size of the input volume given to the encoder
total_train_input = 400 # total input volume
total_test_input = 20 # input for testing the network [10 volumes]
step_for_saving_graph = 50

def interpolationBetnLatentSpace(z1, z2, save_path):
    # -----------interpolation with formula [new_z = (1 - t) * z1 + t * z2] --------------------------
    maximum = 1
    minimum = 0
    interpolated_points = np.linspace(minimum, maximum, 11)

    for t in interpolated_points:

        new_z1 = np.multiply(z1, (1-t))
        new_z2 = np.multiply(z2, t)
        new_z = np.add(new_z1, new_z2)
        #print("new z shape before decoder", new_z.shape, type(new_z))
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)
        variables_names = [s.name for s in tf.trainable_variables()]
        values = sess.run(variables_names)
        file = open("variables_after_restoring.txt", "a+")
        for k, v in zip(variables_names, values):
            file.write("Variables:{}, Shape:{}, Values:{}".format(k, (v.shape), v))

        train_interpol_output = sess.run([ae_outputs], feed_dict={l_space: new_z})
        out = np.reshape(train_interpol_output, (OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_SIZE)).astype(np.float32)
        for_plot.plot_output(out, OUTPUT_SIZE, t)
        if (t == 0):
            print("this is the output shape of decoder after interpolation", out[1])

    return new_z


def resize_batch(imgs):
    # A function to resize a batch of shape images to (32, 32, 32)
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
    for i in range(1, (total_train_input + 1)):
        v = np.loadtxt('/home/gigl/Research/simple_autoencoder/train_data/MyTestFile' + str(i) + '.txt')
        image_matrix = np.reshape(v, (32, 32, 32)).astype(np.float32)
        ax_z, ax_x, ax_y = image_matrix.nonzero()
        input_fig = plt.figure()
        ax = input_fig.add_subplot(111, projection='3d')
        ax.scatter(ax_x, ax_y, ax_z, zdir='z', c='red')
        plt.savefig('input_data/demo' + str(i) + '.png')
        plt.close()
        input_file = np.append(input_file, image_matrix)

    input_file = np.reshape(input_file, (total_train_input, 32 * 32 * 32)).astype(np.float32)
    return input_file


def next_batch(next_batch_array, batchsize, offset):
    rowStart = offset * batchsize
    rowEnd = (rowStart + batchsize)
    return next_batch_array[rowStart:rowEnd, :]

# -------- loss functions from Generative and Discriminative VOXEL modeling paper ------

def weighted_binary_crossentropy(output, target):
    return -(80.0 * target * tf.log(output) + 20 * (1.0 - target) * tf.log(1.0 - output)) / 100.0


# --------------------read data set----------------------
input_file = loadfile()  # load 400 chairs as volume with shape [400,32768]
print("input file shape:", input_file.shape, input_file[0])
tmp = np.reshape(input_file[0], (32,32,32))
for_plot.plot_output(tmp,OUTPUT_SIZE,"inputObjCheck")

# ----------------- calculate the number of batches per epoch --------------------
batch_per_ep = input_file.shape[0] // batch_size  # batch per epoch will be 10 [input total= 400 divided by batch-size = 10 ]

ae_inputs = tf.placeholder(tf.float32, (None, 32, 32, 32, 1), name="encoder_input")  # input to the network (shape volumes)

# ---------for variational auto encoder(this has to be commented when simple auto encoder model is used) --------------
#z_mean, z_std, l_space = md.encoder(ae_inputs)

# ---------for simple auto encoder(this has to be commented when variational model is used) --------------
l_space = md.encoder(ae_inputs)

# --------- Output from decoder ---------------------
ae_outputs = md.decoder(l_space)

# ----------------- calculate the loss and optimize variational auto encoder network ------------------------

#generation_loss = -tf.reduce_sum(ae_inputs * tf.log(1e-8 + ae_outputs) + (1-ae_inputs) * tf.log(1e-8 + 1 - ae_outputs), 1)

#latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_std) - tf.log(tf.square(z_std)) - 1,1)



# Voxel-Wise Reconstruction Loss
# Note that the output values are clipped to prevent the BCE from evaluating log(0).
'''ae_outputs = tf.clip_by_value(ae_outputs, 1e-8, 1 - 1e-8)
bce_loss = tf.reduce_sum(weighted_binary_crossentropy(ae_outputs, ae_inputs), [1,2])
bce_loss = tf.reduce_mean(bce_loss)
# KL Divergence from isotropic gaussian prior
kl_div = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_std) - tf.log(1e-8 + tf.square(z_std)) - 1, [1])
kl_div = tf.reduce_mean(kl_div)

loss = kl_div + bce_loss

optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)'''


# ----------------- calculate the loss and optimize the simple auto encoder network -----------------------------

loss = tf.losses.mean_squared_error(ae_inputs, ae_outputs)  # calculate the mean square error loss
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
bce_loss = tf.constant([1.0])
kl_div = tf.constant([1.0])
# -----------------------initialize the network---------------------------------
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()
    plot_loss = np.zeros([1, 4])

    for ep in range(epoch_num):  # epochs loop
        next_batch_array = input_file  # copy of input file to use for fetching next batch from input array
        for batch_n in range(batch_per_ep):  # batches loop
            batch_img = next_batch(next_batch_array, batch_size, batch_n)  # read a batch
            batch_img = resize_batch(batch_img)  # reshape the images to (32, 32, 32)
            _, c, v_loss, k_loss = sess.run([optimizer, loss, bce_loss, kl_div], feed_dict={ae_inputs: batch_img})
            print('Epoch: {} - cost= {:.5f}'.format((ep + 1), c))
            plot_loss = np.append(plot_loss, [[(ep+1), c, v_loss, k_loss]], axis=0)

    save_path = saver.save(sess, '/home/gigl/Research/simple_autoencoder/checkpoints/model.ckpt')
    print("the model checkpoints save path is %s" % save_path)
    variables_names = [v.name for v in tf.trainable_variables()]
    values = sess.run(variables_names)
    file = open("variables_after_training.txt", "a+")
    for k, v in zip(variables_names, values):
        file.write("Variables:{}, Shape:{}, Values:{}".format(k, (v.shape), v))

    # --------------------plot loss -------------------------------------

    plot_loss = plot_loss[1:, 0:]  # to eliminate first row as it represents 0 epoch and 0 loss
    plt.plot(plot_loss[:, 0], plot_loss[:, 1], c='blue')
    plt.savefig('output_data/total_loss.png')
    plt.close()
    plt.plot(plot_loss[:, 0], plot_loss[:, 2], c='blue')
    plt.savefig('output_data/bce_train_loss.png')
    plt.close()
    plt.plot(plot_loss[:, 0], plot_loss[:, 3], c='blue')
    plt.savefig('output_data/KL_train_loss.png')
    plt.close()

    # ------------------test the trained network for test shapes -------------------------------
    plot_loss_test = np.zeros([1, 2])
    for i in range(1, (total_test_input + 1)):
        temp = np.loadtxt('/home/gigl/Research/simple_autoencoder/test_data/TestFile' + str(i) + '.txt')
        test_img = np.reshape(temp, (32, 32, 32)).astype(np.float32)
        ax_z, ax_x, ax_y = test_img.nonzero()
        input_fig = plt.figure()
        ax = input_fig.add_subplot(111, projection='3d')
        ax.scatter(ax_x, ax_y, ax_z, zdir='z', c='red')
        plt.savefig('test_input_data/demo' + str(i) + '.png')
        plt.close()
        test_img = np.reshape(test_img, (1, 32 * 32 * 32)).astype(np.float32)

        batch_img = resize_batch(test_img)
        _, recon_img, c = sess.run([l_space, ae_outputs, loss], feed_dict={ae_inputs: batch_img})
        out = np.reshape(recon_img, (OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_SIZE)).astype(np.float32)
        print('test shape: {} - cost= {:.5f}'.format(i, c))
        plot_loss_test = np.append(plot_loss_test, [[i, c]], axis=0)

        # -------------------------- plot the reconstructed images -------------------------
        for_plot.plot_output(out, OUTPUT_SIZE, i)

    plot_loss_test = plot_loss_test[1:, 0:]  # to eliminate first row as it represents 0 epoch and 0 loss
    plt.plot(plot_loss_test[:, 0], plot_loss_test[:, 1], c='blue')
    plt.savefig('output_data/test_loss.png')
    plt.close()
    # ------------------- Linear Interpolation --------------------------------

    train_shape_1 = resize_batch(input_file[0, :])
    train_shape_2 = resize_batch(input_file[1, :])
    train_l_space1, train_output_image1 = sess.run([l_space, ae_outputs], feed_dict={ae_inputs: train_shape_1})
    train_output_image1 = np.reshape(train_output_image1, (OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_SIZE)).astype(np.float32)
    for_plot.plot_output(train_output_image1, OUTPUT_SIZE, 'trainimg')
    train_l_space2, train_output_image2 = sess.run([l_space, ae_outputs], feed_dict={ae_inputs: train_shape_2})
    start = timer()
    #print("This is the output shape of decoder before interpolation", train_output_image1)

    new_z = interpolationBetnLatentSpace(train_l_space1, train_l_space2, save_path)
    print("This is the shape before interpolation", train_output_image1[1])
    interpolation_time = timer() - start
    print("Interpolation took %f seconds:", interpolation_time)
    print("This is the shape of train_l_space1", train_l_space1.shape)
    #meshlab_visualize.meshlab_output()
