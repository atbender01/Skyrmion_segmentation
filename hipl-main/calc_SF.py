"""
Functions for generating python static structure factors
"""
import time
import os

import numba
import numpy as np
from numba import jit, cuda
from skimage import measure
import matplotlib.pyplot as plt


def SF(data, Qnum=4, res=200, jit=True, precompile=True, cuda=False):
    """calculate structure factor directly in Fourier space

    Args:
        data (ndarray): [[y1, x1], [y2,x2], ...] position data
        Qnum (float): Define the reciprocal space area for SSF map
                      (from -Qnum*pi to Qnum*pi). Defaults to 4.
        res (int, optional): Output resolution of SF map (res,res). Defaults to 200.
        norm (float, optional): define real space lattice periodicity. Defaults to 2*np.pi.

    Returns:
        ndarray: 2D structure factor map
    """
    data = np.array(data)
    n = len(data)
    q = np.linspace(-Qnum * np.pi, Qnum * np.pi, res)
    qx, qy = np.meshgrid(q, q)
    norm = 2 * np.pi
    # defines realspace lattice periodicity. Changing would be confusing
    # when also defining Qnum

    SF_map = np.ones((res, res), dtype=complex)
    otime = time.time()
    stime = time.time()
    if cuda:
        # print('precompiling?')
        # _SF_cuda[32,32](np.copy(SF_map), data[:4], qx, qy, norm)
        # print('now for real')
        # otime=time.time()
        # _SF_cuda[32,32](SF_map, data, qx, qy, norm)
        print("not implemented yet")
        return

    elif jit:
        if precompile:
            print("precompiling")
            _ = _SF_jit(SF_map, data[:4], qx, qy, norm)
            otime = time.time()
        print(f"running compiled with numba for {len(data)} points")
        SF_map = _SF_jit(SF_map, data, qx, qy, norm, v=1)
        print("100 %")
    else:
        print(f"{0:04.1f}%", end=" .. ")
        for i in range(n):
            if time.time() - stime > 60:
                print(f"{(i/n)*100:04.1f}%", end=" . ")
                stime = time.time()

            y1, x1 = data[i]
            for j in range(n):
                if i != j:
                    y2, x2 = data[j]
                    SF_map += np.exp(
                        -1j * (qx * (x2 - x1) + qy * (y2 - y1)) * (1 / norm)
                    )
        print(f"\n{100:04.1f}%")

    SF_map /= n
    SF_map = SF_map.real  # imaginary part is 0 anyways

    ttime = time.time() - otime
    m, s = divmod(ttime, 60)
    h, m = divmod(m, 60)
    print(f"Completed in {h:02.0f}:{m:02.0f}:{s:02.0f}.{round(s%1*10):01.0f}")

    return SF_map


### works with parallel = False, gives slightly different/wrong answer with parallel=True
@jit(nopython=True, parallel=False)
def _SF_jit(SF_map, data, qx, qy, norm, v=0):
    """Helper function for running SF() with JIT compilation and in parallel using numba.
    Oddly, it's faster to do two full loops of prange(n) rather than prange(i+1, n) for
    the second loop, and having np.exp(r2-r1) + np.exp(r1-r2) all at once. Not sure why
    this is...
    """
    n = len(data)
    for i in numba.prange(n):
        if v and i % 20 == 0:
            print(round(i / n * 100, 1), "%")
        y1, x1 = data[i]
        for j in range(n):
            if i != j:
                y2, x2 = data[j]
                SF_map += np.exp(-1j * (qx * (x2 - x1) + qy * (y2 - y1)) * (1 / norm))

    return SF_map


# @cuda.jit
# def _SF_cuda(SF_map, data, qx, qy, norm):
#     n = len(data)
#     for i in numba.prange(n):
#         y1, x1 = data[i]
#         for j in numba.prange(n):
#             if i != j:
#                 y2, x2 = data[j]
#                 SF_map += np.exp(-1j * (qx * (x2 - x1) + qy * (y2 - y1)) * (1 / norm))


def pdf2D(points, rad=None, scale=1, square=False, point_size=1, binfac=True):
    """
    Gets the pair distribution function of a set of points.
    To calculate the structure factor, take a FFT of this PDF (probably after
    windowing it with a Tukey window).

    points: (N,2) list of points. [[y1,x1], [y2,x2], ... , [yn,xn]]
    rad: radius to look, default dim/8
    scale: scale up the positions of the points by this factor
    square: force dimy=dimx
    """
    points = np.array(points)
    if scale != 1:
        if isinstance(scale, int):
            points *= scale
            if rad is not None:
                rad *= scale
        else:
            points = points.astype("float")
            points *= scale
            if rad is not None:
                rad *= scale

    points = np.around(points).astype("int")
    # points = np.floor(points).astype('int')
    # print('using floor')

    maxs = np.max(points, axis=0)
    mins = np.min(points, axis=0)
    if len(maxs) != 2:
        return "Only 2D input allowed"

    points -= mins
    dimy, dimx = np.ceil(1 + maxs - mins).astype("int")
    if square:
        if dimy > dimx:
            dimx = dimy
        else:
            dimy = dimx
    print("dy, dx: ", dimy, dimx)

    if rad is None:
        rad = int(np.ceil(min(dimy, dimx) / 8))
    else:
        rad = round(rad)
    print(f"Continuing with rad: {rad}")

    rho = len(points) / (dimy * dimx)
    print(f"Density of points after scaling: {rho:g} points/pixel")

    pim = np.zeros((dimy, dimx))
    if point_size == 1:
        points2 = (points[:, 0], points[:, 1])
        pim[points2] = 1
    else:
        for cy, cx in points:
            plus = int(np.ceil(point_size / 2))
            minus = int(np.floor(point_size / 2))
            pim[cy - minus : cy + plus, cx - minus : cx + plus] = 1

    ys = (points[:, 0] >= rad) & (points[:, 0] < dimy - rad)
    xs = (points[:, 1] >= rad) & (points[:, 1] < dimx - rad)
    centers = points[ys & xs]
    ncents = len(centers)
    print(f"Calculating PDF for {ncents} centers")

    pdf = np.zeros((2 * rad, 2 * rad))
    for y, x in centers:
        pdf += pim[y - rad : y + rad, x - rad : x + rad]

    pdf[rad, rad] = 0  # remove aligned peak
    if binfac and scale > 1:
        if isinstance(binfac, (float, int)):
            binfac = round(binfac * scale)
        else:
            binfac = scale
        print("binning")
        pdf = measure.block_reduce(pdf, (binfac, binfac), np.sum)
    return pdf


def sixfold_grid(dy=None, dx=None, sep=1, num_centers=None):
    """
    an ideal sixfold lattice of dimensions dy, dx, with lattice spacing sep
    can optionally define the number of centers
    """
    if num_centers is not None:
        num_x = int(np.sqrt(num_centers) - 2)
        num_y = int(np.sqrt(num_centers) / np.sqrt(3))
    else:
        num_y = round((dy / np.sqrt(3)) / sep)
        num_x = round(dx / sep)

    y1 = np.arange(num_y) * sep * np.sqrt(3)
    x1 = np.arange(num_x) * sep
    X1, Y1 = np.meshgrid(x1, y1)
    xy1 = np.dstack([X1.flatten(), Y1.flatten()]).squeeze()

    y2 = (np.arange(num_y) * np.sqrt(3) + np.sqrt(3) / 2) * sep
    x2 = (np.arange(num_x) + 0.5) * sep
    X2, Y2 = np.meshgrid(x2, y2)
    xy2 = np.dstack([X2.flatten(), Y2.flatten()]).squeeze()

    xy = np.vstack([xy1, xy2])
    hexgrid_yx = xy[:, ::-1]
    return hexgrid_yx


def fourfold_grid(dy=None, dx=None, sep=1, num_centers=None):
    """
    an ideal fourfold lattice of dimensions dy, dx, with lattice spacing sep
    can optionally define the number of centers
    """
    if num_centers is not None:
        num_x = num_y = int(np.sqrt(num_centers))
    else:
        num_y = round(dy / sep)
        num_x = round(dx / sep)

    y1 = np.arange(num_y) * sep + sep // 2
    x1 = np.arange(num_x) * sep + sep // 2
    X1, Y1 = np.meshgrid(x1, y1)
    xy = np.dstack([X1.flatten(), Y1.flatten()]).squeeze()
    squaregrid_yx = xy[:, ::-1]
    return squaregrid_yx


def show_SF(im, Qnum=4, label_every=2, cmap="magma", log=False, save="", **kwargs):
    """Show structure factor with axes labels

    Args:
        im (ndarray): Structure factor (SF) to show
        Qnum (int, optional): Range of F space used in SF. Defaults to 4.
        label_every (int, optional): Print labels every x*pi. Defaults to 2.
        cmap (str, optional): colormap to use. Defaults to 'magma'.
        log (bool, optional): Display on log scale. Defaults to False.
        save (str, optional): Savepath + filename
    """
    from image_helpers import overwrite_rename

    dimy, dimx = im.shape
    assert dimy == dimx

    num_ticks = Qnum * 2 // label_every + 1
    tick_locs = np.linspace(-Qnum, Qnum, num_ticks)
    tick_locs_pix = np.linspace(0, dimx, num_ticks)
    tick_labels = [rf"${str(i)}\pi$" for i in tick_locs.astype("int")]

    _fig, ax = plt.subplots()
    if log:
        nonzeros = np.nonzero(im)
        im[nonzeros] = np.log10(np.abs(im[nonzeros]))
        img = ax.matshow(im, cmap=cmap)
    else:
        img = ax.matshow(im, cmap=cmap)
    ax.xaxis.tick_bottom()
    ax.set_xticks(tick_locs_pix)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_locs_pix)
    ax.set_yticklabels(tick_labels)

    plt.colorbar(img, pad=0.02, format="%.2g")

    if save:
        _root, ext = os.path.splitext(save)
        if not ext:
            save += ".png"
        save = overwrite_rename(save)
        print("saving: ", save)
        plt.savefig(save, dpi=400, bbox_inches="tight")

    plt.show()
