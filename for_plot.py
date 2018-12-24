

import sys
import shutil
import subprocess
from config import cfg, cfg_from_list
from voxel import voxel2obj


def cmd_exists(cmd):
    return shutil.which(cmd) is not None

def plotOutput(voxel_prediction):
    '''Main demo function'''

    # Save prediction into a file named 'prediction.obj' or the given argument
    pred_file_name = sys.argv[1] if len(sys.argv) > 1 else 'prediction.obj'

    # Save the prediction to an OBJ file (mesh file).
    voxel2obj(pred_file_name, voxel_prediction[0, :, :, :, 0] > cfg.TEST.VOXEL_THRESH)


    # Use meshlab or other mesh viewers to visualize the prediction.
    # For Ubuntu>=14.04, you can install meshlab using
    # `sudo apt-get install meshlab`
    if cmd_exists('meshlab'):
        proc1 = subprocess.Popen(['meshlab', pred_file_name])
        #subprocess.Popen.kill(proc1)
        #subprocess.Popen.kill(proc2)
    else:
        print('Meshlab not found: please use visualization of your choice to view %s' %pred_file_name)

