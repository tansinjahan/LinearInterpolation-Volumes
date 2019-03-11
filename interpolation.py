import test_autoencoder as ta
import for_plot
import numpy as np
import tensorflow as tf
from timeit import default_timer as timer


def interpolationBetnLatentSpace(z1,z2,save_path):
    # -----------interpolation with formula [new_z = (1 - t) * z1 + t * z2] --------------------------
    maximum = 1
    minimum = 0
    interpolated_points = np.linspace(minimum, maximum, 12)

    for t in interpolated_points:

        new_z1 = np.multiply(z1, (1-t))
        new_z2 = np.multiply(z2, t)
        new_z = np.add(new_z1, new_z2)
        print("new z shape before decoder", new_z.shape, type(new_z))
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(save_path + '.meta')
            saver.restore(sess, save_path)
            variables_names = [s.name for s in tf.trainable_variables()]
            values = sess.run(variables_names)
            file = open("variables_after_restoring.txt", "a+")
            for k, v in zip(variables_names, values):
                file.write("Variables:{}, Shape:{}, Values:{}".format(k, (v.shape), v))
            if(t == 0):
                print("This is latent vector of training shape 1 after interpolation", new_z)
            train_interpol_output = sess.run([ta.ae_outputs], feed_dict={ta.l_space: new_z})
            out = np.reshape(train_interpol_output, (ta.OUTPUT_SIZE, ta.OUTPUT_SIZE, ta.OUTPUT_SIZE)).astype(np.float32)
            for_plot.plot_output(out, ta.OUTPUT_SIZE, t)
            if (t == 0):
                print("this is the output shape after interpolation", out)
    return new_z

train_shape_1 = ta.resize_batch(ta.input_file[0, :])
train_shape_2 = ta.resize_batch(ta.input_file[1, :])
with tf.Session() as sess:
    train_l_space1, train_output_image1 = sess.run([ta.l_space, ta.ae_outputs], feed_dict={ta.ae_inputs: train_shape_1})
    train_output_image1 = np.reshape(train_output_image1, (ta.OUTPUT_SIZE, ta.OUTPUT_SIZE, ta.OUTPUT_SIZE)).astype(np.float32)
    for_plot.plot_output(train_output_image1, ta.OUTPUT_SIZE, 'trainimg')
    train_l_space2, train_output_image2 = sess.run([ta.l_space, ta.ae_outputs], feed_dict={ta.ae_inputs: train_shape_2})
    start = timer()
    print("This is the output of decoder before interpolation", train_l_space1)
    save_path = '/home/gigl/Research/simple_autoencoder/checkpoints/model.ckpt'
    new_z = interpolationBetnLatentSpace(train_l_space1, train_l_space2, save_path)
    print("This is the shape before interpolation", train_output_image1)
    interpolation_time = timer() - start
    print("Interpolation took %f seconds:", interpolation_time)
    print("This is the shape of train_l_space1", train_l_space1.shape)

