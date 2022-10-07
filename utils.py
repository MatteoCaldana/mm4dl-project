# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


def loo(y_true, y_pred):
    return tf.reduce_max(tf.abs(y_true - y_pred))


def print_err(e):
    print("----------------------")
    err_loo = np.max(e)
    print("err_loo:", err_loo)
    err_mae = np.sum(e) / e.size
    print("err_mae:", err_mae)
    err_mse = np.sqrt(np.sum(e * e) / e.size)
    print("err_mse:", err_mse)
    print("----------------------")
