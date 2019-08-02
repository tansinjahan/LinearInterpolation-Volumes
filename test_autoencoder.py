from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
import pandas as pd
from timeit import default_timer as timer
from mpl_toolkits.mplot3d import Axes3D
import meshlab_visualize
import model as md
import for_plot
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)

print("root:", ROOT_DIR)

# --------------- Define parameters -----------------------------
batch_size = 10  # Number of samples in each batch
epoch_num = 3  # Number of epochs to train the network
lr = 0.001  # Learning rate
OUTPUT_SIZE = 32 # size of the output volume produced by decoder
INPUT_SIZE = 32 # size of the input volume given to the encoder
total_train_input = 400 # total input volume
total_test_input = 20 # input for testing the network [10 volumes]
step_for_saving_graph = 50
dim_of_z = 64

def interpolationBetnLatentSpace(z1, z2, save_path):
    # -----------interpolation with formula [new_z = (1 - t) * z1 + t * z2] --------------------------
    maximum = 1
    minimum = 0
    interpolated_points = np.linspace(minimum, maximum, 11)

    for t in interpolated_points:

        new_z1 = np.multiply(z1, (1-t))
        new_z2 = np.multiply(z2, t)
        new_z = np.add(new_z1, new_z2)
        print("interpolation save path", save_path)
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)
        variables_names = [s.name for s in tf.trainable_variables()]
        values = sess.run(variables_names)
        file = open("variables_after_restoring.txt", "w+")
        for k, v in zip(variables_names, values):
            file.write("Variables:{}, Shape:{}, Values:{}".format(k, (v.shape), v))
        file.close()
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

    #input_file_dic = {}
    imgs = imgs.reshape((-1, 32, 32, 32, 1))

    resized_imgs = np.zeros((imgs.shape[0], 32, 32, 32, 1))
    for i in range(imgs.shape[0]):
        resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (32, 32, 32))

    #print("Input file dictionary info", len(input_file_dic.keys()), len(input_file_dic.values()))
    #print("This is resized image shape", resized_imgs.shape, resized_imgs)
    #dic_evaluation(input_file_dic)
    return resized_imgs


def loadfile():

    input_file = np.array([])
    for i in range(1, (total_train_input + 1)):
        v = np.loadtxt(ROOT_DIR + '/simple_autoencoder/train_data/MyTestFile' + str(i) + '.txt')
        image_matrix = np.reshape(v, (32, 32, 32)).astype(np.float32)

        ax_z, ax_x, ax_y = image_matrix.nonzero()
        input_fig = plt.figure()
        ax = input_fig.add_subplot(111, projection='3d')
        ax.scatter(ax_x, ax_y, ax_z, zdir='z', c='red')
        plt.savefig('input_data_train/demo' + str(i) + '.png')
        plt.close()
        input_file = np.append(input_file, image_matrix)
        

    input_file = np.reshape(input_file, (total_train_input, 32 * 32 * 32)).astype(np.float32)
    return input_file

def dic_evaluation(input_dic):
    for k, v in input_dic.items():
        print("This is volume:{} and it's values:{}".format(k, v))


def next_batch(next_batch_array, batchsize, offset):
    rowStart = offset * batchsize
    rowEnd = (rowStart + batchsize)

    return next_batch_array[rowStart:rowEnd, :]

# -------- loss functions from Generative and Discriminative VOXEL modeling paper ------

def weighted_binary_crossentropy(output, target):
    return -(80.0 * target * tf.log(output) + 20 * (1.0 - target) * tf.log(1.0 - output)) / 100.0


# --------------------read data set----------------------
input_file = loadfile()  # load 400 chairs as volume with shape [400,32768]


# ----------------- calculate the number of batches per epoch --------------------
batch_per_ep = input_file.shape[0] // batch_size  # batch per epoch will be 40 [input total= 400 / 10 ]

ae_inputs = tf.placeholder(tf.float32, (None, 32, 32, 32, 1), name="encoder_input")  # input to the network
#dicForShape = tf.placeholder(tf.string, shape=None, name="volume_name")

# ---------for variational auto encoder(this has to be commented when simple auto encoder model is used) --------------
#z_mean, z_std, l_space = md.encoder(ae_inputs)

# ---------for simple auto encoder(this has to be commented when variational model is used) --------------
l_space = md.encoder(ae_inputs,dim_of_z)

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

            _, z_vec, c, v_loss, k_loss = sess.run([optimizer, l_space, loss, bce_loss, kl_div], feed_dict={ae_inputs: batch_img})
            print('Epoch: {} - cost= {:.5f}'.format((ep + 1), c))
            plot_loss = np.append(plot_loss, [[(ep+1), c, v_loss, k_loss]], axis=0)

    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    save_path = saver.save(sess, ROOT_DIR + '/simple_autoencoder/checkpoints/model.ckpt')
    print("the model checkpoints save path is %s" % save_path)
    variables_names = [v.name for v in tf.trainable_variables()]
    values = sess.run(variables_names)
    file = open("variables_after_training.txt", "w+")
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
    arr = np.zeros([1, dim_of_z])
    file = open("Z_for_trainingShapes.txt", "w+")
    for i in range(1, (total_test_input + 1)):
        temp = np.loadtxt(ROOT_DIR + '/simple_autoencoder/test_data/MyTestFile' + str(i) + '.txt')
        test_img = np.reshape(temp, (32, 32, 32)).astype(np.float32)
        ax_z, ax_x, ax_y = test_img.nonzero()
        input_fig = plt.figure()
        ax = input_fig.add_subplot(111, projection='3d')
        ax.scatter(ax_x, ax_y, ax_z, zdir='z', c='red')
        plt.savefig('input_data_test/demo' + str(i) + '.png')
        plt.close()
        test_img = np.reshape(test_img, (1, 32 * 32 * 32)).astype(np.float32)

        batch_img = resize_batch(test_img)
        z_vec, recon_img, c = sess.run([l_space, ae_outputs, loss], feed_dict={ae_inputs: batch_img})
        for j in range(0, z_vec.shape[0]):
            file.write("for image:{} \n batch, shape, z_vector:{}{}\n{} \n".format(j, batch_img.shape, z_vec.shape, z_vec[j:, :]))
        arr = np.append(arr, z_vec, axis=0)
        out = np.reshape(recon_img, (OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_SIZE)).astype(np.float32)
        print('test shape: {} - cost= {:.5f}'.format(i, c))
        plot_loss_test = np.append(plot_loss_test, [[i, c]], axis=0)

        # -------------------------- plot the reconstructed images -------------------------
        for_plot.plot_output(out, OUTPUT_SIZE, i)

    arr = arr[1:, :]
    print("this is arr", arr, arr.shape)
    for i in range(1, (arr.shape[1] + 1)):
        arr_to_col = np.column_stack(arr[:, :i])
    print(arr_to_col)

    dataset_z = pd.DataFrame(arr_to_col)
    #dataset_z.to_csv('dataset.csv')
    dataset_z.to_excel('z_vector.xlsx')
    print("dataset_z info", dataset_z.shape, dataset_z.head())
    plot_loss_test = plot_loss_test[1:, 0:]  # to eliminate first row as it represents 0 epoch and 0 loss
    plt.plot(plot_loss_test[:, 0], plot_loss_test[:, 1], c='blue')
    plt.savefig('output_data/test_loss.png')
    plt.close()


    # ------------------- Linear Interpolation --------------------------------

    '''train_shape_1 = resize_batch(input_file[0, :])
    train_shape_2 = resize_batch(input_file[1, :])
    train_l_space1, train_output_image1 = sess.run([l_space, ae_outputs], feed_dict={ae_inputs: train_shape_1})
    print("lspace shape", train_l_space1.shape, train_output_image1.shape)
    train_output_image1 = np.reshape(train_output_image1, (OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_SIZE)).astype(np.float32)
    for_plot.plot_output(train_output_image1, OUTPUT_SIZE, 'trainimg')
    train_l_space2, train_output_image2 = sess.run([l_space, ae_outputs], feed_dict={ae_inputs: train_shape_2})
    print("lspace", train_l_space2.shape)
    print(train_l_space2)
    start = timer()
    #print("This is the output shape of decoder before interpolation", train_output_image1)

    new_z = interpolationBetnLatentSpace(train_l_space1, train_l_space2, save_path)
    #print("This is the shape before interpolation", train_output_image1[1])
    interpolation_time = timer() - start
    #print("Interpolation took %f seconds:", interpolation_time)
    #print("This is the shape of train_l_space1", train_l_space1.shape)
    meshlab_visualize.meshlab_output()'''
