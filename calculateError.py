import numpy as np
import tensorflow as tf
import pandas as pd
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error


def reconstruction_error(gt, pred):
    np_err = np.mean((pred - gt)**2)
    print("using Numpy mean square error for reconstruction", np_err)
    sk_err = mean_squared_error(gt, pred)
    print("using Sklearn mean square error for reconstruction", sk_err)

def true_positive():
    return 0

def true_negative():
    return 0

def cal_all_error(gt, pd):
    target = gt
    prediction = pd
    reconstruction_error(target, prediction)


#g = np.array([[[[1,2], [3,4], [9,10]]],[[[1,2], [3,4], [9,10]]]])
#g = np.random.rand(400, 32768)
#print("g shape", g.shape)
#p = np.random.rand(400, 32768)
#p = np.array([[[[5,6],[7,8], [11, 12]]], [[[5,6],[7,8], [11, 12]]]])
#cal_all_error(g, p)