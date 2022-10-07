# -*- coding: utf-8 -*-
import os
import re
import numpy as np
from scipy import io as sio

from geomdl import NURBS


def parseline(line):
    return np.array([float(x) for x in line.split(" ") if x != ""])


def read_nurbs_mesh(filename):
    with open(filename, "r") as fp:
        data = fp.read()
    data = data.splitlines()
    assert len(data) == 13
    return {
        "knots": [parseline(data[8]), parseline(data[9])],
        "ctrlptsw": np.array(
            [parseline(data[10]), parseline(data[11]), parseline(data[12])]
        ).T,
    }


def get_degree(knots):
    d = 1
    while knots[d] == 0:
        d += 1
    assert knots[-d] == 1 and knots[-(d + 1)] != 1
    return d - 1


def build_nurbs_txt(mesh):
    srf = NURBS.Surface()
    srf.degree_u = get_degree(mesh["knots"][0])
    srf.degree_v = get_degree(mesh["knots"][1])
    nu = mesh["knots"][0].size - srf.degree_u - 1
    nv = mesh["knots"][1].size - srf.degree_v - 1
    idxs = np.arange(nv) * nu
    idxs = np.concatenate([idxs + i for i in range(nu)])
    srf.set_ctrlpts(
        [[*x[:2], 0, x[-1]] for x in mesh["ctrlptsw"][idxs]], nu, nv,
    )
    srf.knotvector_u = mesh["knots"][0]
    srf.knotvector_v = mesh["knots"][1]
    return srf


def build_nurbs_mat(g_nurbs):
    srf = NURBS.Surface()
    srf.degree_u = g_nurbs["order"][0, 0][0, 0] - 1
    srf.degree_v = g_nurbs["order"][0, 0][0, 1] - 1
    nu = g_nurbs["number"][0, 0][0, 0]
    nv = g_nurbs["number"][0, 0][0, 1]
    coefs = g_nurbs["coefs"][0, 0].reshape((4, -1)).T
    srf.set_ctrlpts(
        [[*x] for x in coefs], nu, nv,
    )
    srf.knotvector_u = g_nurbs["knots"][0, 0][0, 0][0]
    srf.knotvector_v = g_nurbs["knots"][0, 0][0, 1][0]
    return srf


def load_data_old(filename, plot=False, test=True):
    data = sio.loadmat(filename)
    srf = build_nurbs_mat(data["g_nurbs"])
    x, y = [np.ravel(data[f], order="F") for f in ["X", "Y"]]
    xy = np.vstack([x, y]).T
    u, v = [np.ravel(f) for f in np.meshgrid(*data["vtk_pts"][0])]
    uv = np.vstack([u, v]).T

    if test or plot:
        pts = np.array(srf.evaluate_list(uv))
        err = np.abs(xy - pts[:, :2])
        max_err = np.max(err)
        print("Error:", max_err)
        if max_err > 1e-15:
            raise Exception("Test not passed.")

    if plot:
        srf.vis = vis.VisSurface()
        srf.render()

        plt.figure()
        plt.scatter(pts[:, 0], pts[:, 1])
        plt.scatter(x, y, marker="x")
        plt.figure()
        plt.contourf(data["X"], data["Y"], data["eu"])
    return srf, data, uv, xy, np.ravel(data["eu"], order="F")


def struct_to_np(struct):
    return np.concatenate([struct[f][0, 0] for f in np.sort(struct.dtype.names)])


def load_data(filename, structured, plot=False, test=True):
    data = sio.loadmat(filename)

    xyz = data["F"].reshape(data["F"].shape[0], -1, order="F").T
    u, v = [np.ravel(f) for f in np.meshgrid(*data["vtk_pts"][0])]
    uv = np.vstack([u, v]).T

    if test or plot:
        srf = build_nurbs_mat(data["g_nurbs"])
        pts = np.array(srf.evaluate_list(uv))
        err = np.abs(xyz - pts[:, : xyz.shape[1]])
        max_err = np.max(err)
        print("Error:", max_err)
        if max_err > 1e-15:
            raise Exception("Test not passed.")

    if plot:
        srf.vis = vis.VisSurface()
        srf.render()

    if structured:
        return (
            struct_to_np(data["params"]),
            data["F"],
            data["eu"],
        )
    else:
        return (
            struct_to_np(data["params"]),
            uv,
            xyz,
            np.reshape(data["eu"], (-1, uv.shape[0]), order="F").T,
        )


def load_list(folder, pattern, structured, test=False):
    ls = os.listdir(folder)
    ls = [filename for filename in ls if re.match(pattern, filename)]

    rs = []
    ps = []
    for filename in ls:
        print("Loading:", filename)
        if structured:
            p, xyz, f = load_data(
                f"{folder}/{filename}", structured, plot=False, test=test
            )
            if len(f.shape) == 3:
                f = np.moveaxis(f, [0, 1, 2], [2, 0, 1])
            rs.append(f)
            ps.append(p)
        else:
            p, uv, xyz, f = load_data(
                f"{folder}/{filename}", structured, plot=False, test=test
            )
            p = np.repeat(p, uv.shape[0], axis=1).T
            rs.append(np.hstack([uv, p, xyz, f]))

    if structured:
        return np.hstack(ps).T, np.stack(rs)
    else:
        return np.vstack(rs)


###############################################################################
# TESTS
###############################################################################


def test_txt():
    m = read_nurbs_mesh(r"./data/laplacian/geo_quarter_ring.txt")
    srf = build_nurbs_txt(m)
    srf.vis = vis.VisSurface()
    srf.render()


def test_old():
    srf, data, uv, xy, u = load_data_old(r"./data/laplacian/Square_Test.mat", plot=True)
    plt.figure()
    plt.scatter(xy[:, 0], xy[:, 1], c=u)


def test1():
    p, uv, xy, f = load_data(
        r"./data/laplacian/geo_quarter_ring_a0000b0500_out_127.mat",
        plot=True,
        test=True,
    )
    plt.figure()
    plt.scatter(xy[:, 0], xy[:, 1], c=f)


def test2():
    p, uv, xy, f = load_data(
        r"./data/kirchoff-love-scordelis-lo/kirchoff_lovel_scrodelis_lo_t0500l2000_out_64.mat",
        plot=True,
        test=True,
    )
    plt.figure()
    plt.scatter(xy[:, 0], xy[:, 1], c=f)


def test_load_list():
    r1 = load_list(
        r"./data/laplacian/",
        r"geo_quarter_ring_a[0-9]+b[0-9]+_out_127.mat",
        structured=False,
    )
    r2 = load_list(
        r"./data/kirchoff-love-scordelis-lo/",
        r"kirchoff_lovel_scrodelis_lo_t[0-9]+l[0-9]+_out_64.mat",
        structured=True,
    )


if __name__ == "__main__":
    from geomdl.visualization import VisMPL as vis
    import matplotlib.pyplot as plt

    plt.close("all")

    # test_txt()
    # test_old()
    # test1()
    # test2()
    test_load_list()
