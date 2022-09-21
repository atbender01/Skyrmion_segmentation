import io
import os
import sys
import textwrap
from itertools import takewhile

import numpy as np


def load_ovf(file=None, sim="norm", B0=1e4, v=1):
    """ Load a .ovf or .omf file of magnetization values.

    This function takes magnetization output files from OOMMF or Mumax, pulls
    some data from the header and returns 3D arrays for each magnetization
    component as well as the pixel resolutions.

    Args:
        file (string): Path to file
        sim (string): Define how the magnetization is scaled as it's read from
            the file. OOMMF writes .omf files with vectors in units of A/m,
            while Mumax writes .omf files with vectors normalized. This allows
            the reading to scale the vectors appropriately to gauss or simply
            make sure everything is normalized (as is needed for the phase
            calculation).

            - "OOMMF": Vectors scaled by mu0 and output in Tesla
            - "mumax": Vectors scaled by B0 and given those units (gauss or T)
            - "norm": (default) Normalize all vectors (does not change (0,0,0) vectors)
            - "raw": Don't do anything with the values.

        B0 (float): Saturation induction (gauss). Only relevant if sim=="mumax"
        v (int): Verbosity.

            - 0 : No output
            - 1 : Default output
            - 2 : Extended output, print full header.

    Returns:
        tuple: (mag_x, mag_y, mag_z, del_px)

        - mag_x (`2D array`) -- x-component of magnetization (units depend on `sim`).
        - mag_y (`2D array`) -- y-component of magnetization (units depend on `sim`).
        - mag_z (`2D array`) -- z-component of magnetization (units depend on `sim`).
        - del_px (float) -- Scale of datafile in y/x direction (nm/pixel)
        - zscale (float) -- Scale of datafile in z-direction (nm/pixel)
    """
    vprint = print if v >= 1 else lambda *a, **k: None

    with io.open(file, mode="r") as f:
        try:
            header = list(takewhile(lambda s: s[0] == "#", f))
        except UnicodeDecodeError:  # happens with binary files
            header = []
            with io.open(file, mode="rb") as f2:
                for line in f2:
                    if line.startswith("#".encode()):
                        header.append(line.decode())
                    else:
                        break
    if v >= 2:
        ext = os.path.splitext(file)[1]
        print(f"-----Start {ext} Header:-----")
        print("".join(header).strip())
        print(f"------End {ext} Header:------")

    dtype = None
    header_length = 0
    for line in header:
        header_length += len(line)
        if line.startswith("# xnodes"):
            xsize = int(line.split(":", 1)[1])
        if line.startswith("# ynodes"):
            ysize = int(line.split(":", 1)[1])
        if line.startswith("# znodes"):
            zsize = int(line.split(":", 1)[1])
        if line.startswith("# xstepsize"):
            xscale = float(line.split(":", 1)[1])
        if line.startswith("# ystepsize"):
            yscale = float(line.split(":", 1)[1])
        if line.startswith("# zstepsize"):
            zscale = float(line.split(":", 1)[1])
        if line.startswith("# Begin: Data Text"):
            vprint("Text file found")
            dtype = "text"
        if line.startswith("# Begin: Data Binary 4"):
            vprint("Binary 4 file found")
            dtype = "bin4"
        if line.startswith("# Begin: Data Binary 8"):
            vprint("Binary 8 file found")
            dtype = "bin8"

    if xsize is None or ysize is None or zsize is None:
        print(
            textwrap.dedent(
                f"""\
    Simulation dimensions are not given. \
    Expects keywords "xnodes", "ynodes, "znodes" for number of cells.
    Currently found size (x y z): ({xsize}, {ysize}, {zsize})"""
            )
        )
        sys.exit(1)
    else:
        vprint(f"Simulation size (z, y, x) : ({zsize}, {ysize}, {xsize})")

    if xscale is None or yscale is None or zscale is None:
        vprint(
            textwrap.dedent(
                f"""\
    Simulation scale not given. \
    Expects keywords "xstepsize", "ystepsize, "zstepsize" for scale (nm/pixel).
    Found scales (z, y, x): ({zscale}, {yscale}, {xscale})"""
            )
        )
        del_px = np.max([i for i in [xscale, yscale, 0] if i is not None]) * 1e9
        if zscale is None:
            zscale = del_px
        else:
            zscale *= 1e9
        vprint(
            f"Proceeding with {del_px:.3g} nm/pixel for in-plane and \
            {zscale:.3g} nm/pixel for out-of-plane."
        )
    else:
        assert xscale == yscale
        del_px = xscale * 1e9  # originally given in meters
        zscale *= 1e9
        if zscale != del_px:
            vprint(f"Image (x-y) scale : {del_px:.3g} nm/pixel.")
            vprint(f"Out-of-plane (z) scale : {zscale:.3g} nm/pixel.")
        else:
            vprint(f"Image scale : {del_px:.3g} nm/pixel.")

    if dtype == "text":
        data = np.genfromtxt(file)  # takes care of comments automatically
    elif dtype == "bin4":
        # for binaries it has to give count or else will take comments at end as well
        data = np.fromfile(
            file, dtype="f", count=xsize * ysize * zsize * 3, offset=header_length + 4
        )
    elif dtype == "bin8":
        data = np.fromfile(
            file, dtype="f", count=xsize * ysize * zsize * 3, offset=header_length + 8
        )
    else:
        print("Unkown datatype given. Exiting.")
        sys.exit(1)

    data = data.reshape(
        (zsize, ysize, xsize, 3)
    )  # binary data not always shaped nicely

    if sim.lower() == "oommf":
        vprint("Scaling for OOMMF datafile.")
        mu0 = 4 * np.pi * 1e-7  # output in Tesla
        data *= mu0
    elif sim.lower() == "mumax":
        vprint(f"Scaling for mumax datafile with B0={B0:.3g}.")
        data *= B0  # output is same units as B0
    elif sim.lower() == "raw":
        vprint("Not scaling datafile.")
    elif sim.lower() == "norm":
        data = data.reshape((zsize * ysize * xsize, 3))  # to iterate through vectors
        row_sums = np.sqrt(np.sum(data ** 2, axis=1))
        rs2 = np.where(row_sums == 0, 1, row_sums)
        data = data / rs2[:, np.newaxis]
        data = data.reshape((zsize, ysize, xsize, 3))
    else:
        print(
            textwrap.dedent(
                """\
        Improper argument given for sim. Please set to one of the following options:
            'oommf' : vector values given in A/m, will be scaled by mu0
            'mumax' : vectors all of magnitude 1, will be scaled by B0
            'raw'   : vectors will not be scaled."""
            )
        )
        sys.exit(1)

    mag_x = data[:, :, :, 0]
    mag_y = data[:, :, :, 1]
    mag_z = data[:, :, :, 2]

    return (mag_x, mag_y, mag_z, del_px, zscale)
