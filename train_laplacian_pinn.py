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


def build_model_bottleneck(layers, layers_map, kws):
    inputs = tf.keras.Input(shape=(layers[0],))
    x = inputs
    for layer in layers_map:
        x = tf.keras.layers.Dense(layer, **kws)(x)
    u = tf.keras.layers.Dense(2)(x)

    y = tf.keras.layers.Concatenate(axis=1)([inputs, u])
    for layer in layers[1:-1]:
        y = tf.keras.layers.Dense(layer, **kws)(y)
    outputs = tf.keras.layers.Dense(layers[-1])(y)
    return tf.keras.Model(inputs=inputs, outputs=[u, outputs])


def build_model(layers, kws):
    act = kws["activation"] if kws["activation"] != "prelu" else None
    inputs = tf.keras.Input(shape=(layers[0],))
    x = inputs
    for layer in layers[1:-1]:
        x = tf.keras.layers.Dense(layer, **{**kws, "activation": act})(x)
        if act is None:
            x = keras.layers.PReLU()(x)
    outputs = tf.keras.layers.Dense(layers[-1])(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def plot_model(plot, uvabxyf, xi, model):
    if plot is not None:
        fig, axs = plt.subplots(len(plot), 3, figsize=(13, 10))

        axs = axs.reshape((len(plot), 3))
        for i, p in enumerate(plot):
            idxs = np.intersect1d(
                np.where(uvabxyf[:, 2] == p[0])[0], np.where(uvabxyf[:, 3] == p[1])[0]
            )
            xy = uvabxyf[idxs, 4:6]
            xx = xy[:, 0].reshape((127, 127))
            yy = xy[:, 1].reshape((127, 127))
            f = uvabxyf[idxs, -1].reshape((127, 127))
            fp = model.predict(uvabxyf[idxs, xi[0] : xi[1]]).reshape((127, 127))
            err = np.abs(fp - f)

            im = axs[i, 0].contourf(xx, yy, f)
            plt.colorbar(im, ax=axs[i, 0])
            axs[i, 0].set_title(f"FOM solution $\mu=({p[0]},{p[1]})$")
            axs[i, 0].set_ylabel("Y")
            #
            im = axs[i, 1].contourf(xx, yy, fp)
            plt.colorbar(im, ax=axs[i, 1])
            axs[i, 1].set_title("UC-USM-Net solution")
            #
            im = axs[i, 2].contourf(xx, yy, err, norm=matplotlib.colors.LogNorm())
            plt.colorbar(im, ax=axs[i, 2])
            axs[i, 2].set_title("Error")

            if i == len(plot) - 1:
                for j in range(3):
                    axs[i, j].set_xlabel("X")

        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        plt.tight_layout()
        plt.savefig("lapl_pinn_trained.pdf")


def main_test_simple(
    xi_kind,
    depth=7,
    width=60,
    activation="elu",
    subsample_size=10 ** 4,
    plot=None,
    eval_err=True,
):
    LOCALS = {**locals()}
    TABLEXI = {"param": (0, 4), "physic": (2, 6)}
    xi = TABLEXI[xi_kind]

    uvabxyf = read_nurbs.load_list(
        r"./data/laplacian/",
        r"geo_quarter_ring_a[0-9]+b[0-9]+_out_127.mat",
        structured=False,
        test=False,
    )
    idxs = np.random.choice(uvabxyf.shape[0], subsample_size, replace=False)

    model = build_model(
        [4] + [width] * depth + [1],
        {
            "activation": activation,
            "kernel_initializer": "glorot_uniform",
            "bias_initializer": "zeros",
        },
    )

    callback = keras.callbacks.ReduceLROnPlateau(
        factor=0.5, patience=10, monitor="loss", min_delta=1e-10
    )
    model.compile(optimizer="adam", loss="mse", metrics=[loo])

    keras.backend.set_value(model.optimizer.learning_rate, 0.001)
    t0 = time.time()
    model.fit(
        uvabxyf[idxs, xi[0] : xi[1]],
        uvabxyf[idxs, -1],
        epochs=300,
        batch_size=32,
        shuffle=True,
        validation_split=0.0,
        callbacks=[callback],
        verbose=2,
    )
    print("Total training time:", time.time() - t0, "[s]")

    f = uvabxyf[:, -1]

    plot_model(plot, uvabxyf, xi, model)

    model.summary()
    print(LOCALS)
    if eval_err:
        fp = model.predict(uvabxyf[:, xi[0] : xi[1]]).reshape((-1,))
        err = np.abs(fp - f)
        print_err(err)


def main_test_bottleneck(
    layers_pre, layers_post, activation="elu", subsample_size=10 ** 4
):
    LOCALS = {**locals()}
    uvabxyf = read_nurbs.load_list(
        r"./data/laplacian/",
        r"geo_quarter_ring_a[0-9]+b[0-9]+_out_127.mat",
        structured=False,
        test=False,
    )
    idxs = np.random.choice(uvabxyf.shape[0], subsample_size, replace=False)

    model = build_model_bottleneck(
        [4] + layers_pre + [1],
        layers_post,
        {
            "activation": activation,
            "kernel_initializer": "glorot_uniform",
            "bias_initializer": "zeros",
        },
    )

    callback = keras.callbacks.ReduceLROnPlateau(
        factor=0.5, patience=10, monitor="loss", min_delta=1e-10
    )
    model.compile(optimizer="adam", loss="mse", metrics=[loo])

    keras.backend.set_value(model.optimizer.learning_rate, 0.001)
    t0 = time.time()
    model.fit(
        uvabxyf[idxs, 2:6],
        [uvabxyf[idxs, :2], uvabxyf[idxs, -1]],
        epochs=300,
        batch_size=32,
        shuffle=False,
        validation_split=0.0,
        callbacks=[callback],
        verbose=2,
    )
    print("Total training time:", time.time() - t0, "[s]")

    up = model.predict(uvabxyf[:, 2:6])[1].reshape((-1,))
    model.summary()
    print(LOCALS)
    err = np.abs(up - uvabxyf[:, -1])
    print_err(err)


if __name__ == "__main__":
    main_test_simple(
        "param",
        depth=8,
        width=80,
        activation="elu",
        subsample_size=4 * 10 ** 4,
        plot=[[0, 3], [0.5, 1], [1, 1.5]],
        eval_err=False,
    )
