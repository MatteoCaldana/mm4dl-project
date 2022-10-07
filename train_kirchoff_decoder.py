# -*- coding: utf-8 -*-
import numpy as np
import time

import tensorflow as tf
from tensorflow import keras

import read_nurbs
from utils import loo, print_err
from train_laplacian_decoder import build_model

np.random.seed(0)
tf.random.set_seed(0)


def load_data():
    params, f = read_nurbs.load_list(
        r"./data/kirchoff-love-scordelis-lo/",
        r"kirchoff_lovel_scrodelis_lo_t[1-3][0-9]+l[2-4][0-9]+_out_64_v2.mat",
        structured=True,
        test=False,
    )
    params = params[:, [1, 4], None, None]
    mmin, mmax = np.min(f, axis=(0, 1, 2)), np.max(f, axis=(0, 1, 2))
    f = (f - mmin) / (mmax - mmin)
    f = f[..., None]
    print("SHAPE", params.shape, f.shape)
    return params, f


def main_test(train_frac, kernel_size, activation, filters, batch_size, strides):
    LOCALS = {**locals()}

    params, f = load_data()
    idxs = np.random.choice(
        params.shape[0], int(params.shape[0] * train_frac), replace=False
    )

    model = build_model(
        kernel_size=kernel_size,
        strides=strides,
        activation=activation,
        filters=filters,
        out_filters=3,
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
        epochs=200,
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


if __name__ == "__main__":
    main_test(
        train_frac=0.8,
        kernel_size=7,
        activation="prelu",
        filters=40,
        batch_size=32,
        strides=[2, 1, 2, 1, 2, 1, 2, 1, 2],
    )

    main_test(
        train_frac=0.8,
        kernel_size=9,
        activation="prelu",
        filters=72,
        batch_size=32,
        strides=[2, 1, 2, 1, 2, 1, 2, 1, 2],
    )

    for (ks, f) in [
        (7, 40),
        (7, 72),
        (9, 40),
        (9, 72),
        (11, 40),
        (11, 64),
    ]:
        main_test(
            train_frac=0.8,
            kernel_size=ks,
            activation="prelu",
            filters=f,
            batch_size=16,
            strides=[2, 1, 2, 1, 2, 1, 2, 1, 2],
        )
