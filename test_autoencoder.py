from __future__ import division, print_function, absolute_import
import tensorflow.contrib.layers as lays

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import meshlab_visualize
from skimage import transform
import model as md
from skimage import measure
from mpl_toolkits.mplot3d import Axes3D
from timeit import default_timer as timer


# --------------- Define parameters -----------------------------
batch_size = 10  # Number of samples in each batch
epoch_num = 50  # Number of epochs to train the network
lr = 0.001  # Learning rate
OUTPUT_SIZE = 32 # size of the output volume produced by decoder
INPUT_SIZE = 32 # size of the input volume given to the encoder
total_train_input = 400 # total input volume
total_test_input = 10 # input for testing the network [10 volumes]
step_for_saving_graph = 50

def interpolationBetnLatentSpace(z1, z2, save_path):
    # -----------interpolation with formula [new_z = (1 - t) * z1 + t * z2] --------------------------
    maximum = 1
    minimum = 0
    interpolated_points = np.array([0, 0.5, 0.8])
    # np.linspace(minimum, maximum, 11)

    for t in interpolated_points:

        new_z1 = np.multiply(z1, (1-t))
        new_z2 = np.multiply(z2, t)
        new_z = np.add(new_z1, new_z2)
        print("new z shape before decoder", new_z.shape, type(new_z))
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path + '.meta')
        if(t == 0):
            print("This is latent vecor of training shape 1 after interpolation", new_z)
        train_interpol_output = sess.run([an_outputs], feed_dict={decoder_Z_input: new_z})
        out = np.reshape(train_interpol_output, (OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_SIZE)).astype(np.float32)
        plot_output(out, OUTPUT_SIZE, t)
        if (t == 0):
            print("this is the interpolated volume", out)

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
    #print("This is the shape of input for 100 shape", input_file.shape)
    return input_file

def next_batch(next_batch_array, batchsize, offset):
    rowStart = offset * batchsize
    rowEnd = (rowStart + batchsize) - 1
    return next_batch_array[rowStart:rowEnd, :]

def plot_output(out_array, OUTPUT_SIZE, filename):
    plotOutArr = np.array([])
    for x_i in range(0, OUTPUT_SIZE):
        for y_j in range(0, OUTPUT_SIZE):
            for z_k in range(0, OUTPUT_SIZE):
                if out_array[x_i, y_j, z_k] > 0.5:
                    plotOutArr = np.append(plotOutArr, 1)
                else:
                    plotOutArr = np.append(plotOutArr, 0)

    output_image = np.reshape(plotOutArr, (OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_SIZE)).astype(np.float32)

    # Use marching cubes to obtain the surface mesh of these volumes
    verts, faces, normals, values = measure.marching_cubes_lewiner(output_image, 0)
    faces = faces + 1
    for_save = open('output_data/test_volume' + str(filename) + '.obj', 'w')
    for item in verts:
        for_save.write("v {0} {1} {2}\n".format(item[0], item[1], item[2]))

    for item in normals:
        for_save.write("vn {0} {1} {2}\n".format(item[0], item[1], item[2]))

    for item in faces:
        for_save.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(item[0], item[1], item[2]))

    for_save.close()

    z, x, y = output_image.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, zdir='z', c='red')
    plt.savefig('output_data/test_volume' + str(filename) + '.png')
    plt.close()
    for_text_save = np.reshape(output_image, (OUTPUT_SIZE * OUTPUT_SIZE * OUTPUT_SIZE))
    np.savetxt('output_data/test_volume' + str(filename) + '.txt', for_text_save)


# --------------------read data set----------------------
input_file = loadfile()  # load 100 chairs as volume with shape [100,32768]

# ----------------- calculate the number of batches per epoch --------------------
batch_per_ep = input_file.shape[0] // batch_size  # batch per epoch will be 10 [input total=100 divided by batch-size = 10 ]

ae_inputs = tf.placeholder(tf.float32, (None, 32, 32, 32, 1))  # input to the network (shape volumes)
decoder_Z_input = tf.placeholder(tf.float32, (None, 2, 2, 2, 8))

l_space = md.encoder(ae_inputs)
ae_outputs = md.decoder(l_space)
an_outputs = md.decoder(decoder_Z_input)

# ----------------- calculate the loss and optimize the network--------------------------------
loss = tf.reduce_mean(tf.square(ae_outputs - ae_inputs))  # calculate the mean square error loss
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# -----------------------initialize the network---------------------------------
init = tf.global_variables_initializer()
saver = tf.train.Saver()

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

    plot_loss = plot_loss[1:, 0:] # to eliminate first row as it represents 0 epoch and 0 loss
    plt.plot(plot_loss[:, 0], plot_loss[:, 1], c='blue')
    plt.savefig('output_data/test_loss.png')
    plt.close()

    # ------------------test the trained network for test shapes -------------------------------
    #firstTestShape_Zvector =
    for i in range(1, (total_test_input + 1)):
        temp = np.loadtxt('/home/gigl/Research/simple_autoencoder/test_data/TestImg' + str(i) + '.txt')
        test_img = np.reshape(temp, (32, 32, 32)).astype(np.float32)
        ax_z, ax_x, ax_y = test_img.nonzero()
        input_fig = plt.figure()
        ax = input_fig.add_subplot(111, projection='3d')
        ax.scatter(ax_x, ax_y, ax_z, zdir='z', c='red')
        plt.savefig('test_input_data/demo' + str(i) + '.png')
        plt.close()
        test_img = np.reshape(test_img, (1, 32 * 32 * 32)).astype(np.float32)

        batch_img = resize_batch(test_img)
        recon_img = sess.run([l_space, ae_outputs], feed_dict={ae_inputs: batch_img})[1]
        out = np.reshape(recon_img, (OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_SIZE)).astype(np.float32)
        # print("this is the shape of output volume", out.shape)

        # -------------------------- plot the reconstructed images -------------------------
        plot_output(out, OUTPUT_SIZE, i)

    save_path = saver.save(sess, '/home/gigl/Research/simple_autoencoder/checkpoints/model.ckpt')
    print("the model checkpoints save path is %s" % save_path)
    #------------------- Linear Interpolation --------------------------------

    train_shape_1 = resize_batch(input_file[0, :])
    train_shape_2 = resize_batch(input_file[1, :])
    train_l_space1, train_output_image1 = sess.run([l_space, ae_outputs], feed_dict={ae_inputs: train_shape_1})
    train_output_image1 = np.reshape(train_output_image1, (OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_SIZE)).astype(np.float32)
    plot_output(train_output_image1, OUTPUT_SIZE, 'trainimg')
    train_l_space2, train_output_image2 = sess.run([l_space, ae_outputs], feed_dict={ae_inputs: train_shape_2})
    start = timer()
    print("This is the output of decoder before interpolation", train_output_image1)
    # print("This is latent vector of training shape 1 before interpolation", train_l_space1)
    meshlab_visualize.meshlab_output()
    new_z = interpolationBetnLatentSpace(train_l_space1, train_l_space2, save_path)

    interpolation_time = timer() - start
    print("Interpolation took %f seconds:", interpolation_time)
    print("This is the shape of train_l_space1", train_l_space1.shape)

