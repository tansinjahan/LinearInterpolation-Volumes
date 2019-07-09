from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

def plot_output(out_array, OUTPUT_SIZE, filename):
    plotOutArr = np.array([])
    with_border_arr = np.array([])
    for x_i in range(0, OUTPUT_SIZE):
        for y_j in range(0, OUTPUT_SIZE):
            for z_k in range(0, OUTPUT_SIZE):
                if out_array[x_i, y_j, z_k] > 0.6:
                    plotOutArr = np.append(plotOutArr, 1)
                else:
                    plotOutArr = np.append(plotOutArr, 0)

    '''for x_i in range(1, OUTPUT_SIZE+1):
        for y_j in range(1, OUTPUT_SIZE+1):
            for z_k in range(1, OUTPUT_SIZE+1):
                 with_border_arr[x_i, y_j, z_k] = plotOutArr[x_i -1, y_j -1, z_k -1]'''

    output_image = np.reshape(plotOutArr, (OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_SIZE)).astype(np.float32)

    # Use marching cubes to obtain the surface mesh of these volumes
    verts, faces, normals, values = measure.marching_cubes_lewiner(output_image, level=0.0, gradient_direction='descent')

    faces = faces + 1
    for_save = open('output_data/test_volume' + str(filename) + '.obj', 'w')
    for item in verts:
        for_save.write("v {0} {1} {2}\n".format(item[0], item[1], item[2]))

    for item in normals:
        for_save.write("vn {0} {1} {2}\n".format(-item[0], -item[1], -item[2]))

    for item in faces:
        for_save.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(item[0], item[2], item[1]))

    for_save.close()

    z, x, y = output_image.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, zdir='z', c='red')
    plt.savefig('output_data/test_volume' + str(filename) + '.png')
    plt.close()
    for_text_save = np.reshape(plotOutArr, (OUTPUT_SIZE * OUTPUT_SIZE * OUTPUT_SIZE))
    np.savetxt('output_data/test_volume' + str(filename) + '.txt', for_text_save)