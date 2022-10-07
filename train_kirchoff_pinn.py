# -*- coding: utf-8 -*-
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib

import tensorflow as tf
from tensorflow import keras

import read_nurbs
from utils import loo, print_err
from train_laplacian_pinn import build_model

np.random.seed(0)
tf.random.set_seed(0)


def plot_model(plot, uvabxyzf, xi, model):
    if plot is not None:
        fig_shape = (len(plot), 6)
        fig, axs = plt.subplots(*fig_shape, figsize=(17, 10))
        axs = axs.reshape(fig_shape)
        for i, p in enumerate(plot):
            idxx = np.where(np.abs(uvabxyzf[:, 2] - p[0]) < 1e-4)[0]
            idxy = np.where(np.abs(uvabxyzf[:, 3] - p[1]) < 1e-4)[0]
            idxs = np.intersect1d(idxx, idxy)
            uu = uvabxyzf[idxs, 0].reshape((64, 64))
            vv = uvabxyzf[idxs, 1].reshape((64, 64))
            f = uvabxyzf[idxs, -3:].reshape((64, 64, 3))
            fp = model.predict(uvabxyzf[idxs, xi[0] : xi[1]]).reshape(f.shape)
            err = np.abs(fp - f)
            
            axs[i, 0].set_ylabel(fr'$\mu=({p[0]},{p[1]:.2f})$'+'\n'+r'$\xi_2$')

            for j in range(3):
                ax0, ax1 = axs[i, 2 * j], axs[i, 2 * j + 1]
                im = ax0.contourf(uu, vv, f[:, :, j])
                plt.colorbar(im, ax=ax0, format='%0.1e')
                im = ax1.contourf(
                    uu, vv, err[:, :, j], norm=matplotlib.colors.LogNorm()
                )
                plt.colorbar(im, ax=ax1)
                
                if i == fig_shape[0] - 1:
                    ax0.set_xlabel(r'$\xi_1$')
                    ax1.set_xlabel(r'$\xi_1$')
                    
                if i == 0:
                    ax0.set_title(fr'FOM displacement $u_{j+1}$')
                    ax1.set_title(fr'Error displacement $|u_{j+1} - \tildeu_{j+1}|$')
                    
        plt.subplots_adjust(wspace=0.2)
        plt.tight_layout()
        plt.savefig("kirchoff_pinn_trained.pdf")


def main_test_simple(
    xi_kind,
    depth=7,
    width=60,
    activation="elu",
    subsample_size=10 ** 4,
    batch_size=32,
    plot=None,
    eval_err=True,
):
    LOCALS = {**locals()}
    TABLEXI = {"param": (0, 4), "physic": (2, 7)}
    xi = TABLEXI[xi_kind]

    uvabxyzf = read_nurbs.load_list(
        r"./data/kirchoff-love-scordelis-lo/",
        r"kirchoff_lovel_scrodelis_lo_t[1-3][0-9]+l[2-4][0-9]+_out_64_v2.mat",
        structured=False,
        test=False,
    )
    f = uvabxyzf[:, -3:]
    mmax, mmin = np.max(f, axis=0), np.min(f, axis=0)
    uvabxyzf[:, -3:] = (f - mmin) / (mmax - mmin)
    uvabxyzf = uvabxyzf[:, [0, 1, 3, 6, 8, 9, 10, 11, 12, 13]]
    print("Data size", uvabxyzf.shape)
    idxs = np.random.choice(uvabxyzf.shape[0], subsample_size, replace=False)

    XX_train, YY_train = uvabxyzf[idxs, xi[0] : xi[1]], uvabxyzf[idxs, -3:]
    XX, YY = uvabxyzf[:, xi[0] : xi[1]], uvabxyzf[:, -3:]

    model = build_model(
        [xi[1] - xi[0]] + [width] * depth + [3],
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
    model.summary()

    keras.backend.set_value(model.optimizer.learning_rate, 0.001)
    t0 = time.time()
    model.fit(
        XX_train,
        YY_train,
        epochs=200,
        batch_size=batch_size,
        shuffle=False,
        validation_split=0.0,
        callbacks=[callback],
        verbose=2,
    )
    print("Total training time:", time.time() - t0, "[s]")
    model.summary()
    print(LOCALS)

    plot_model(plot, uvabxyzf, xi, model)

    if eval_err:
        fp = model.predict(XX, verbose=2)
        err = np.abs(fp.reshape(YY.shape) - YY)
        print_err(err)


if __name__ == "__main__":
    main_test_simple(
        "param",
        depth=10,
        width=100,
        activation="elu",
        subsample_size=32 * 10 ** 4,
        batch_size=128,
        plot=[[25, 0.33300882], [49, 0.33300882], [38, 0.76026542], [25, 1.23778751], [49, 1.23778751]],
        eval_err=False
    )
