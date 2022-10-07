# -*- coding: utf-8 -*-
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib

import tensorflow as tf
from tensorflow import keras

import read_nurbs
from utils import loo, print_err

np.random.seed(0)
tf.random.set_seed(0)


def build_model(kernel_size, strides, activation, filters, out_filters=1):
    activation_argument = None if activation == "prelu" else activation
    inputs = keras.layers.Input(shape=(2, 1, 1))

    x = keras.layers.Conv2DTranspose(
        filters,
        (kernel_size, kernel_size),
        strides=(1, 2),
        activation=activation_argument,
        kernel_initializer="he_uniform",
        padding="same",
    )(inputs)
    if activation == "prelu":
        x = keras.layers.PReLU()(x)
    for s in strides:
        x = keras.layers.Conv2DTranspose(
            filters,
            (kernel_size, kernel_size),
            strides=s,
            activation=activation_argument,
            kernel_initializer="he_uniform",
            padding="same",
        )(x)
        if activation == "prelu":
            x = keras.layers.PReLU()(x)
    x = keras.layers.Conv2D(out_filters, (kernel_size, kernel_size), padding="same")(x)

    return keras.Model(inputs, x)


def plot_model(plot, f, fp, err, params):
    if plot is not None:
        ax_shape = (len(plot), 3)
        f_shape = f.shape[1:3]
        fig, axs = plt.subplots(*ax_shape)
        axs = axs.reshape(ax_shape)
        xx, yy = np.meshgrid(
            np.linspace(0, 1, f.shape[1]), np.linspace(0, 1, f.shape[2])
        )
        for i, idx in enumerate(plot):
            print(params[idx, ...].reshape((-1, )))
            im = axs[i, 0].contourf(xx, yy, f[idx, :, :].reshape(f_shape))
            plt.colorbar(im, ax=axs[i, 0])
            im = axs[i, 1].contourf(xx, yy, fp[idx, :, :].reshape(f_shape))
            plt.colorbar(im, ax=axs[i, 1])
            im = axs[i, 2].contourf(
                xx, yy, err[idx, :, :].reshape(f_shape), norm=matplotlib.colors.LogNorm()
            )
            plt.colorbar(im, ax=axs[i, 2])


def load_data():
    params, f = read_nurbs.load_list(
        r"./data/laplacian/",
        r"geo_quarter_ring_a[0-9]+b[0-9]+_out_127.mat",
        structured=True,
        test=False,
    )
    params = params[..., None, None]
    f = f[:, ::2, ::2, None]
    return params, f


def main_test(subsample_size, filters, kernel_size, activation, batch_size, plot=None):
    LOCALS = {**locals()}
    params, f = load_data()
    idxs = np.random.choice(params.shape[0], subsample_size, replace=False)

    model = build_model(
        kernel_size=kernel_size,
        strides=[2, 2, 2, 2, 2],
        activation=activation,
        filters=filters,
    )

    callback = keras.callbacks.ReduceLROnPlateau(
        factor=0.5, patience=10, monitor="loss", min_delta=1e-10
    )
    model.compile(optimizer="adam", loss="mse", metrics=[loo])

    keras.backend.set_value(model.optimizer.learning_rate, 0.001)
    t0 = time.time()
    model.fit(
        params[idxs, ...],
        f[idxs, ...],
        epochs=100,
        batch_size=batch_size,
        shuffle=True,
        validation_split=0.0,
        callbacks=[callback],
        verbose=2,
    )
    print("Total training time:", time.time() - t0, "[s]")

    yp = model.predict(params).reshape(f.shape)
    model.summary()
    print(LOCALS)
    err = np.abs(yp - f)
    print_err(err)
    axs = (1, 2, 3)
    err_eps = np.mean(np.sqrt(np.sum(err * err, axis=axs) / np.sum(f * f, axis=axs)))
    print("----------------------")
    print("err_eps:", err_eps)
    print("----------------------")

    plot_model(plot, f, yp, err, params)


if __name__ == "__main__":
    sims = [
        {"filters": 64, "kernel_size": 4, "activation": "prelu", "batch_size": 32,},
    ]

    for o in sims:
        main_test(subsample_size=180, **o, plot=[14, 105, 210])
