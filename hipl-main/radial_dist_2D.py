import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import time
from multiprocessing import Pool
from scipy.spatial import cKDTree


def shift_points(points, pad=0):
    points[:, 1] += -1 * np.min(points[:, 1]) + pad
    points[:, 0] += -1 * np.min(points[:, 0]) + pad
    return points


def get_dist(pos1, pos2):
    squared = (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2
    return np.sqrt(squared)


def plot_RDF(
    r,
    Gr,
    vlines=None,
    scale=None,
    maxrad=None,
    minrad=0,
    title=None,
    save=None,
    loglog=False,
    x_label=None,
    ax=None,
    **kwargs,
):
    """
    if plotting loglog, will plot G(r) - 1 as in
    Bernard & KrauthPhys. Rev. Lett. 107, (2011).
    maxrad in nm
    """

    if ax is None:
        _fig, ax = plt.subplots()

    if x_label is not None:
        pass
    elif scale is None:
        scale = 1
        x_label = "pixels"
    else:
        x_label = "nm"

    if maxrad is not None:
        if scale is None:
            scale = 1
        else:
            if r[-1] * scale > maxrad:
                ind = np.argmax(r * scale > maxrad)
            else:
                ind = -1
                maxrad = r[-1] * scale

        r = r[:ind]
        Gr = Gr[:ind]
    else:
        maxrad = r[-1] * scale

    if loglog:
        if round(Gr[0]) == 1:
            Gr = Gr - 1
        ax.loglog(r * scale, Gr, **kwargs)
        ax.set_ylim([1e-2, 10])
    else:
        ax.plot(r * scale, Gr, **kwargs)

    if vlines is not None:
        ax.vlines(vlines, ymin=0, ymax=Gr.max(), color="r")

    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$g(r)$")
    ax.set_xbound([minrad, maxrad])

    if title is not None:
        plt.title(title)
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.show()


def plot_RDF_RG6(Tilt_Im, vlines=None, maxrad=None, title=None, save=None):
    fig, ax1 = plt.subplots()
    r, Gr = Tilt_Im.rGr
    r6, Gr6 = Tilt_Im.rGr6
    scale = Tilt_Im.scale

    if maxrad is not None:
        if r[-1] * scale > maxrad:
            ind = np.argmax(r * scale > maxrad)
        else:
            ind = -1
        if r6[-1] * scale > maxrad:
            ind6 = np.argmax(r6 * scale > maxrad)
        else:
            ind6 = -1

        r = r[:ind]
        Gr = Gr[:ind]
        r6 = r6[:ind6]
        Gr6 = Gr6[:ind6]

    ax1.plot(r, Gr, c="b", label=r"$G(r)$")
    ax2 = ax1.twinx()
    ax2.plot(r6, Gr6, c="r", label=r"$G_6(r)$")

    if vlines is not None:
        ax1.vlines(vlines, ymin=0, ymax=Gr.max(), color="r")

    if scale is None:
        x_label = "pixels"
    else:

        def mjrFormatter(x, pos):
            return f"{scale*x:.3g}"

        x_label = "nm"
        ax1.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter))

    ax1.set_xlabel(x_label)
    ax1.set_ylabel(r"$G(r)$", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax2.set_ylabel(r"$G_6(r)$", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    if title is not None:
        plt.title(title)
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.show()


def RingArea(r1, r2, point, bbox):
    """
    area of ring with outer rad r2, inner rad r1, center poin point [y,x] contained
    within a box with one corner at [0,0] and the other at [bbox[0], bbox[1]]

    """
    y, x = point
    bounds = [[-y, bbox[0] - y], [-x, bbox[1] - x]]
    area1 = CircArea(r1, bounds)
    area2 = CircArea(r2, bounds)
    return area2 - area1


def CircArea(rad, bounds):
    area = 0
    for yb in bounds[0]:
        for xb in bounds[1]:
            area += QuadrantArea(rad, np.abs(yb), np.abs(xb))
    return area


def QuadrantArea(rad, yb, xb):
    if rad ** 2 >= yb ** 2 + xb ** 2:  # corner inside circle
        return xb * yb
    elif yb >= rad and xb >= rad:  # circle contained within rectangle
        return np.pi / 4 * rad ** 2
    elif rad >= xb and rad <= yb:
        theta = np.arcsin(xb / rad)
        return (
            xb / 2 * np.sqrt(rad ** 2 - xb ** 2) + rad ** 2 * theta / 2
        )  # triangle area + fraction of circle
    elif rad <= xb and rad >= yb:
        theta = np.arcsin(yb / rad)
        return (
            yb / 2 * np.sqrt(rad ** 2 - yb ** 2) + rad ** 2 * theta / 2
        )  # triangle area + fraction of circle
    elif rad > yb and rad > xb:
        alpha = np.arccos(xb / rad)
        beta = np.arccos(yb / rad)
        wedge = rad ** 2 * (np.pi / 2 - alpha - beta) / 2
        return (
            yb / 2 * np.sqrt(rad ** 2 - yb ** 2)
            + xb / 2 * np.sqrt(rad ** 2 - xb ** 2)
            + wedge
        )
    else:
        print("Houston...")
        return


def RDF2D(points, dr, rad=None, pad=0):
    """
    points : [[y1,x1], [y2,x2], ... ] list of positions
    dr : width of shell in pixels
    rad : maximum radius to use, if None will choose half of corner distance

    TODO:
        - deal with dr not being evenly divisible by rad
        - distance matrix implementation
    """
    num_points = len(points)
    points = np.array(points)
    dim_y = int(np.max(points[:, 0]) - np.min(points[:, 0]) + 2 * pad)
    dim_x = int(np.max(points[:, 1]) - np.min(points[:, 1]) + 2 * pad)
    # recenter points with padding so  0,0 is one corner
    points = shift_points(np.copy(points), pad=pad)
    rho_mean = len(points) / (dim_y * dim_x)

    if rad is None:
        rad = 0.5 * np.sqrt(dim_y ** 2 + dim_x ** 2)

    r = np.arange(0, rad, dr)
    Gr = np.zeros(r.shape)  # empty bins for G(r) final
    nonempty_shells = np.zeros(r.shape)

    i = 0
    print(f"\rRDF calculation: {0}/{num_points}", end="")
    for p1 in points:
        if (i + 1) % 100 == 0:
            print(f"\rRDF calculation: {i+1}/{num_points}", end="")
        j = 0
        localGr = np.zeros(r.shape)
        for p2 in points:
            if i != j:
                dist = get_dist(p1, p2)
                if dist < rad:
                    Idx = int(dist // dr)
                    localGr[Idx] += 1
            j += 1

        # account for shape of ring
        for k in range(len(r)):
            Aring = RingArea(r[k], r[k] + dr, p1, [dim_y, dim_x])
            if Aring > 0:
                localGr[k] = localGr[k] / Aring
                nonempty_shells[k] += 1
        Gr = Gr + localGr
        i += 1

    print(f"\rRDF calculation: {num_points}/{num_points}")
    Gr = Gr / nonempty_shells
    Gr = Gr / rho_mean
    return r + dr / 2, Gr


def Orientation_correlation(skyrm_list, dr, rad=None, pad=0, mode="psi6", mbox=True):
    """
    points : [[y1,x1], [y2,x2], ... ] list of positions
    dr : width of shell in pixels
    rad : maximum radius to use, if None will choose half of corner distance

    """

    points = []
    for skyrm in skyrm_list:
        if skyrm.psi6 is not None and skyrm.psi4 is not None:
            if mode.lower() == "psi6":
                psi = skyrm.psi6
                psic = skyrm.psi6c
            elif mode.lower() == "psi4":
                psi = skyrm.psi4
                psic = skyrm.psi4c
            point = [skyrm.pos[0], skyrm.pos[1], psi, psic]
            if mbox:
                if skyrm.in_mbox:
                    points.append(point)
            else:
                points.append(point)

    points = np.array(points)

    num_points = len(points)
    points = np.array(points)
    dim_y = int((np.max(points[:, 0]) - np.min(points[:, 0]) + 2 * pad).real)
    dim_x = int((np.max(points[:, 1]) - np.min(points[:, 1]) + 2 * pad).real)
    # recenter points with padding so  0,0 is one corner
    points = shift_points(np.copy(points), pad=pad)

    if rad is None:
        rad = 0.5 * np.sqrt(dim_y ** 2 + dim_x ** 2)

    r = np.arange(0, rad, dr)
    numbins = rad // dr

    Gr_inds = np.zeros((len(r), 3)).astype("complex")  # empty bins for G(r) final

    i = 0
    print(f"\rRG6 calculation: 0/{num_points}", end="")
    for p1 in points:
        if (i + 1) % 100 == 0:
            print(f"\rRG6 calculation: {i+1}/{num_points}", end="")
        j = 0
        for p2 in points:
            if i != j:
                dist = get_dist(p1[:2], p2[:2]).real
                if dist < rad:
                    Idx = int(dist // dr)
                    corr = p1[2] * np.conj(p2[2])
                    corrc = p1[3] * np.conj(p2[3])
                    Gr_inds[Idx, 0] += 1
                    Gr_inds[Idx, 1] += corr
                    Gr_inds[Idx, 2] += corrc
            j += 1
        i += 1
    print(f"\rRG6 calculation: {num_points}/{num_points}")

    # zinds = np.where(Gr_inds[:,1]==0)
    # print("Using 1 minimum")
    zinds = np.where(Gr_inds[:, 0] <= 10)
    print("Using 10 minimum")

    Gr_inds[:, 0][zinds] = 1
    Gr = Gr_inds[:, 1] / Gr_inds[:, 0]
    Grc = Gr_inds[:, 2] / Gr_inds[:, 0]

    Gr[zinds] = 0
    Grc[zinds] = 0

    assert np.max(np.abs(np.imag(Gr))) < 1e-5  # the imaginary parts should cancel
    return r + dr / 2, Gr.real, Grc.real


# from https://github.com/by256/rdfpy


def paralell_hist_loop(
    radii_and_indices, kdtree, particles, mins, maxs, N_radii, dr, eps, rho
):
    """RDF histogram loop process for multiprocessing"""
    N, d = particles.shape
    g_r_partial = np.zeros(shape=(N_radii))

    for r_idx, r in radii_and_indices:
        r_idx = int(r_idx)
        # find all particles that are at least r + dr away from the edges of the box
        valid_idxs = np.bitwise_and.reduce(
            [
                (particles[:, i] - (r + dr) >= mins[i])
                & (particles[:, i] + (r + dr) <= maxs[i])
                for i in range(d)
            ]
        )
        valid_particles = particles[valid_idxs]

        # compute n_i(r) for valid particles.
        for particle in valid_particles:
            n = kdtree.query_ball_point(
                particle, r + dr - eps, return_length=True
            ) - kdtree.query_ball_point(particle, r, return_length=True)
            g_r_partial[r_idx] += n

        # normalize
        n_valid = len(valid_particles)
        shell_vol = (
            (4 / 3) * np.pi * ((r + dr) ** 3 - r ** 3)
            if d == 3
            else np.pi * ((r + dr) ** 2 - r ** 2)
        )
        g_r_partial[r_idx] /= n_valid * shell_vol * rho

    return g_r_partial


def rdf(particles, dr, rho=None, rcutoff=0.9, eps=1e-15, parallel=True, progress=False):
    """
    Computes 2D or 3D radial distribution function g(r) of a set of particle
    coordinates of shape (N, d). Particle must be placed in a 2D or 3D cuboidal
    box of dimensions [width x height (x depth)].

    Parameters
    ----------
    particles : (N, d) np.array
        Set of particle from which to compute the radial distribution function
        g(r). Must be of shape (N, 2) or (N, 3) for 2D and 3D coordinates
        repsectively.
    dr : float
        Delta r. Determines the spacing between successive radii over which g(r)
        is computed.
    rho : float, optional
        Number density. If left as None, box dimensions will be inferred from
        the particles and the number density will be calculated accordingly.
    rcutoff : float
        radii cutoff value between 0 and 1. The default value of 0.9 means the
        independent variable (radius) over which the RDF is computed will range
        from 0 to 0.9*r_max. This removes the noise that occurs at r values
        close to r_max, due to fewer valid particles available to compute the
        RDF from at these r values.
    eps : float, optional
        Epsilon value used to find particles less than or equal to a distance
        in KDTree.
    parallel : bool, optional
        Option to enable or disable multiprocessing. Enabling this affords
        significant increases in speed.
    progress : bool, optional
        Set to False to disable progress readout (only valid when
        parallel=False).


    Returns
    -------
    g_r : (n_radii) np.array
        radial distribution function values g(r).
    radii : (n_radii) np.array
        radii over which g(r) is computed
    """

    if not isinstance(particles, np.ndarray):
        particles = np.array(particles)
    # assert particles array is correct shape
    shape_err_msg = "particles should be an array of shape N x d, where N is \
                     the number of particles and d is the number of dimensions."
    assert len(particles.shape) == 2, shape_err_msg
    # assert particle coords are 2 or 3 dimensional
    assert particles.shape[-1] in [
        2,
        3,
    ], "RDF can only be computed in 2 or 3 \
                                           dimensions."

    start = time.time()

    mins = np.min(particles, axis=0)
    maxs = np.max(particles, axis=0)
    # translate particles such that the particle with min coords is at origin
    particles = particles - mins

    # dimensions of box
    dims = maxs - mins

    r_max = (np.min(dims) / 2) * rcutoff
    radii = np.arange(dr, r_max, dr)

    N, d = particles.shape
    if not rho:
        rho = N / np.prod(dims)  # number density

    # create a KDTree for fast nearest-neighbor lookup of particles
    tree = cKDTree(particles)

    if parallel:
        N_radii = len(radii)
        radii_and_indices = np.stack([np.arange(N_radii), radii], axis=1)
        radii_splits = np.array_split(radii_and_indices, os.cpu_count(), axis=0)
        values = [
            (radii_splits[i], tree, particles, mins, maxs, N_radii, dr, eps, rho)
            for i in range(len(radii_splits))
        ]
        with Pool() as pool:
            results = pool.starmap(paralell_hist_loop, values)
        g_r = np.sum(results, axis=0)
    else:
        g_r = np.zeros(shape=(len(radii)))
        for r_idx, r in enumerate(radii):
            # find all particles that are at least r + dr away from the edges of the box
            valid_idxs = np.bitwise_and.reduce(
                [
                    (particles[:, i] - (r + dr) >= mins[i])
                    & (particles[:, i] + (r + dr) <= maxs[i])
                    for i in range(d)
                ]
            )
            valid_particles = particles[valid_idxs]

            # compute n_i(r) for valid particles.
            for particle in valid_particles:
                n = tree.query_ball_point(
                    particle, r + dr - eps, return_length=True
                ) - tree.query_ball_point(particle, r, return_length=True)
                g_r[r_idx] += n

            # normalize
            n_valid = len(valid_particles)
            shell_vol = (
                (4 / 3) * np.pi * ((r + dr) ** 3 - r ** 3)
                if d == 3
                else np.pi * ((r + dr) ** 2 - r ** 2)
            )
            g_r[r_idx] /= n_valid * shell_vol * rho

            if progress:
                print(
                    "Computing RDF     Radius {}/{}    Time elapsed: {:.3f} s".format(
                        r_idx + 1, len(radii), time.time() - start
                    ),
                    end="\r",
                    flush=True,
                )

    return g_r, radii
