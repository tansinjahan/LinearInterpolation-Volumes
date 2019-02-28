import numpy as np
import shutil
import subprocess
import glob

def cmd_exists(cmd):
    return shutil.which(cmd) is not None

def meshlab_output():
    path = '/home/gigl/Research/simple_autoencoder/output_data/'
    if cmd_exists('meshlab'):
        filecounter = len(glob.glob1(path,'*.obj'))
        print("Filecounter{}" .format(filecounter))
        for i in range(1, filecounter):
            proc1 = subprocess.Popen(['meshlab', path + 'test_volume' + str(i) + '.obj'])
    else:
        print('Meshlab not found: please use visualization of your choice to view')

