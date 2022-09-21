from scipy.ndimage import rotate
import numpy as np


def rotate_arr(
    arr_x=None, arr_y=None, arr_z=None, theta_x=0, theta_y=0, theta_z=0, round=1e-3
):
    """rotate array of vector components (dim_z, dim_y, dim_x) around x axis then y then z.
    theta_x and theta_y in degrees.
    gives components x,y,z in microscope frame."""
    tx = np.deg2rad(-theta_x)  # - cuz rotates cw otherwise
    ty = np.deg2rad(theta_y)
    tz = np.deg2rad(theta_z)

    Rx = [[1, 0, 0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]]

    Ry = [[np.cos(ty), 0, np.sin(ty)], [0, 1, 0], [-np.sin(ty), 0, np.cos(ty)]]

    Rz = [[np.cos(tz), -np.sin(tz), 0], [np.sin(tz), np.cos(tz), 0], [0, 0, 1]]

    assert arr_x.shape == arr_y.shape == arr_z.shape
    if len(arr_x.shape) == 2:
        arr_x = np.expand_dims(arr_x, 0)
        arr_y = np.expand_dims(arr_y, 0)
        arr_z = np.expand_dims(arr_z, 0)
    dim_z, dim_y, dim_x = arr_x.shape

    rarr_x = np.zeros((dim_z, dim_y, dim_x))
    rarr_y = np.zeros((dim_z, dim_y, dim_x))
    rarr_z = np.zeros((dim_z, dim_y, dim_x))

    # rotating the vectors
    for k in range(dim_z):
        print(f"-{k+1}/{dim_z}-", end="\r")
        for j in range(dim_y):
            for i in range(dim_x):
                vec = (arr_x[k, j, i], arr_y[k, j, i], arr_z[k, j, i])
                rot_vec = np.matmul(Rz, np.matmul(Ry, np.matmul(Rx, vec)))
                rarr_x[k, j, i] = rot_vec[0]
                rarr_y[k, j, i] = rot_vec[1]
                rarr_z[k, j, i] = rot_vec[2]

    if theta_x != 0:
        rarr_x = rotate(rarr_x, angle=theta_x, axes=(0, 1), cval=0)
        rarr_y = rotate(rarr_y, angle=theta_x, axes=(0, 1), cval=0)
        rarr_z = rotate(rarr_z, angle=theta_x, axes=(0, 1), cval=0)
    if theta_y != 0:
        rarr_x = rotate(rarr_x, angle=theta_y, axes=(0, 2), cval=0)
        rarr_y = rotate(rarr_y, angle=theta_y, axes=(0, 2), cval=0)
        rarr_z = rotate(rarr_z, angle=theta_y, axes=(0, 2), cval=0)
    if theta_z != 0:
        theta_z *= -1
        rarr_x = rotate(rarr_x, angle=theta_z, axes=(1, 2), cval=0)
        rarr_y = rotate(rarr_y, angle=theta_z, axes=(1, 2), cval=0)
        rarr_z = rotate(rarr_z, angle=theta_z, axes=(1, 2), cval=0)

    if theta_x == 0 and theta_y == 0:
        rarr_x = np.squeeze(rarr_x)
        rarr_y = np.squeeze(rarr_y)
        rarr_z = np.squeeze(rarr_z)

    if round:
        rarr_mags = np.sqrt(rarr_x ** 2 + rarr_y ** 2 + rarr_z ** 2)
        rarr_mags /= np.max(rarr_mags)
        zeros = np.where(np.abs(1 - rarr_mags) > round)
        rarr_x[zeros] = 0
        rarr_y[zeros] = 0
        rarr_z[zeros] = 0

    return (rarr_x, rarr_y, rarr_z)
