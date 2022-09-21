"""
Helper functions for when working with any image data

Arthur McCray
amccray@anl.gov
"""

import os
import time

from textwrap import dedent

import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    import numba
except ImportError:
    print("Numba not available")

import numpy as np
import skimage
from ipywidgets import interact
from numba import jit
from scipy import ndimage as ndi
from scipy.signal import tukey
from scipy.spatial.transform import Rotation as R

###
### Functions for displaying images
###


def show_im(
    image,
    title=None,
    simple=False,
    origin="upper",
    cbar=True,
    cbar_title="",
    scale=None,
    save=None,
    **kwargs,
):
    """Display an image on a new axis.

    Takes a 2D array and displays the image in grayscale with optional title on
    a new axis. In general it's nice to have things on their own axes, but if
    too many are open it's a good idea to close with plt.close('all').

    Args:
        image (2D array): Image to be displayed.
        title (str): (`optional`) Title of plot.
        simple (bool): (`optional`) Default output or additional labels.

            - True, will just show image.
            - False, (default) will show a colorbar with axes labels, and will adjust the
              contrast range for images with a very small range of values (<1e-12).

        origin (str): (`optional`) Control image orientation.

            - 'upper': (default) (0,0) in upper left corner, y-axis goes down.
            - 'lower': (0,0) in lower left corner, y-axis goes up.

        cbar (bool): (`optional`) Choose to display the colorbar or not. Only matters when
            simple = False.
        cbar_title (str): (`optional`) Title attached to the colorbar (indicating the
            units or significance of the values).
        scale (float): Scale of image in nm/pixel. Axis markers will be given in
            units of nanometers.

    Returns:
        None
    """
    image = np.array(image)
    if image.dtype == "bool":
        image = image.astype("int")
    ndim = np.ndim(image)
    if ndim == 2:
        pass
    elif ndim == 3:
        if image.shape[2] != 3 and image.shape[2] != 4:
            print(
                dedent(
                    """\
                Input image is 3D and does not seem to be a color image.
                Summing along first axis"""
                )
            )
            image = np.sum(image, axis=0)
    else:
        print(f"Input image is of dimension {ndim}. Please input 2D image.")
        return

    if simple and title is None:
        # all this to avoid a white border when saving the image
        fig = plt.figure()
        aspect = image.shape[0] / image.shape[1]
        size = kwargs.get("size", (4, 4 * aspect))
        if isinstance(size, (int, float)):
            size = (size, size)
        fig.set_size_inches(size)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        fig.add_axes(ax)
    else:
        _fig, ax = plt.subplots()

    cmap = kwargs.get("cmap", "gray")
    if simple:
        vmin = kwargs.get("vmin", None)
        vmax = kwargs.get("vmax", None)
    else:
        # adjust coontrast range if minimal range detected
        # avoids people thinking 0 phase shift images (E-15) are real
        vmin = kwargs.get("vmin", np.min(image) - 1e-12)
        vmax = kwargs.get("vmax", np.max(image) + 1e-12)

    im = ax.matshow(image, origin=origin, vmin=vmin, vmax=vmax, cmap=cmap)

    if title is not None:
        ax.set_title(str(title))

    if simple:
        # ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        pass

    else:
        plt.tick_params(axis="x", top=False)
        ax.xaxis.tick_bottom()
        ax.tick_params(direction="in")
        if scale is None:
            ticks_label = "pixels"
        else:

            def mjrFormatter(x, pos):
                return f"{scale*x:.3g}"

            fov = scale * max(image.shape[0], image.shape[1])
            if kwargs.get("scale_units", None) is None:
                if fov < 4e3:  # if fov < 4um use nm scale
                    ticks_label = " nm "
                elif fov > 4e6:  # if fov > 4mm use m scale
                    ticks_label = "  m  "
                    scale /= 1e9
                else:  # if fov between the two, use um
                    ticks_label = r" $\mu$m "
                    scale /= 1e3
            else:
                ticks_label = kwargs.get("scale_units")

            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter))
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter))

        if origin == "lower":
            ax.text(y=0, x=0, s=ticks_label, rotation=-45, va="top", ha="right")
        elif origin == "upper":  # keep label in lower left corner
            ax.text(
                y=image.shape[0], x=0, s=ticks_label, rotation=-45, va="top", ha="right"
            )

        if cbar:
            plt.colorbar(im, ax=ax, pad=0.02, format="%g", label=str(cbar_title))

    if save:
        print("saving: ", save)
        dpi = kwargs.get("dpi", 400)
        if simple and title is None:
            plt.savefig(save, dpi=dpi, bbox_inches=0)
        else:
            plt.savefig(save, dpi=dpi, bbox_inches="tight")

    plt.show()
    return


def show_stack(images, ptie=None, origin="upper", title=False, scale_each=True):
    """Shows a stack of dm3s or np images with a slider to navigate slice axis.

    Uses ipywidgets.interact to allow user to view multiple images on the same
    axis using a slider. There is likely a better way to do this, but this was
    the first one I found that works...

    If a TIE_params object is given, only the regions corresponding to ptie.crop
    will be shown.

    Args:
        images (list): List of 2D arrays. Stack of images to be shown.
        ptie (``TIE_params`` object): Will use ptie.crop to show only the region
            that will remain after being cropped.
        origin (str): (`optional`) Control image orientation.
        title (bool): (`optional`) Try and pull a title from the signal objects.
    Returns:
        None
    """
    images = np.array(images)
    if not scale_each:
        vmin = np.min(images)
        vmax = np.max(images)

    if ptie is None:
        top, bot = 0, images[0].shape[0]
        left, r = 0, images[0].shape[1]
    else:
        if ptie.rotation != 0 or ptie.x_transl != 0 or ptie.y_transl != 0:
            rotate, x_shift, y_shift = ptie.rotation, ptie.x_transl, ptie.y_transl
            for i, _ in enumerate(images):
                images[i] = ndi.rotate(images[i], rotate, reshape=False)
                images[i] = ndi.shift(images[i], (-y_shift, x_shift))
        top = ptie.crop["top"]
        bot = ptie.crop["bottom"]
        left = ptie.crop["left"]
        r = ptie.crop["right"]

    images = images[:, top:bot, left:r]

    _fig, _ax = plt.subplots()
    plt.axis("off")
    N = images.shape[0]

    def view_image(i=0):
        if scale_each:
            _im = plt.imshow(
                images[i], cmap="gray", interpolation="nearest", origin=origin
            )
        else:
            _im = plt.imshow(
                images[i],
                cmap="gray",
                interpolation="nearest",
                origin=origin,
                vmin=vmin,
                vmax=vmax,
            )

        if title:
            plt.title("Stack[{:}]".format(i))

    interact(view_image, i=(0, N - 1))
    return


def show_2D(
    mag_x,
    mag_y,
    mag_z=None,
    a=15,
    l=None,
    w=None,
    title=None,
    color=True,
    hsv=True,
    origin="upper",
    save=None,
    ax=None,
    rad=None,
    **kwargs,
):
    """Display a 2D vector arrow plot.

    Displays an an arrow plot of a vector field, with arrow length scaling with
    vector magnitude. If color=True, a colormap will be displayed under the
    arrow plot.

    If mag_z is included and color=True, a spherical colormap will be used with
    color corresponding to in-plane and white/black to out-of-plane vector
    orientation.

    Args:
        mag_x (2D array): x-component of magnetization.
        mag_y (2D array): y-component of magnetization.
        mag_z (2D array): optional z-component of magnetization.
        a (int): Number of arrows to plot along the x and y axes. Default 15.
        l (float): Scale factor of arrows. Larger l -> shorter arrows. Default None
            guesses at a good value. None uses matplotlib default.
        w (float): Width scaling of arrows. None uses matplotlib default.
        title (str): (`optional`) Title for plot. Default None.
        color (bool): (`optional`) Whether or not to show a colormap underneath
            the arrow plot. Color image is made from colorwheel.color_im().
        hsv (bool): (`optional`) Only relevant if color == True. Whether to use
            an hsv or 4-fold color-wheel in the color image.
        origin (str): (`optional`) Control image orientation.
        save (str): (`optional`) Path to save the figure.

    Returns:
        fig: Returns the figure handle.
    """
    assert mag_x.ndim == mag_y.ndim
    if mag_x.ndim == 3:
        print("Summing along first axis")
        mag_x = np.sum(mag_x, axis=0)
        mag_y = np.sum(mag_y, axis=0)
        if mag_z is not None:
            mag_z = np.sum(mag_z, axis=0)

    if a > 0:
        # a = ((mag_x.shape[0] - 1) // a) + 1
        a = int(((mag_x.shape[0] - 1) / a) + 1)

    dimy, dimx = mag_x.shape
    X = np.arange(0, dimx, 1)
    Y = np.arange(0, dimy, 1)
    U = mag_x
    V = mag_y

    sz_inches = 8
    if color:
        if rad is None:
            rad = mag_x.shape[0] // 16
            rad = max(rad, 16)
            pad = 10  # pixels
            width = np.shape(mag_y)[1] + 2 * rad + pad
            aspect = dimy / width
        elif rad == 0:
            width = np.shape(mag_y)[1]
            aspect = dimy / width
        else:
            pad = 10  # pixels
            width = np.shape(mag_y)[1] + 2 * rad + pad
            aspect = dimy / width
    else:
        aspect = dimy / dimx

    if ax is None:
        fig, ax = plt.subplots()
    ax.set_aspect(aspect)

    if color:
        from colorwheel import color_im

        im = ax.matshow(
            color_im(mag_x, mag_y, mag_z, hsvwheel=hsv, rad=rad),
            cmap="gray",
            origin=origin,
        )
        arrow_color = "white"
        plt.axis("off")
    else:
        arrow_color = "black"

    if a > 0:
        ashift = (dimx - 1) % a // 2
        q = ax.quiver(
            X[ashift::a],
            Y[ashift::a],
            U[ashift::a, ashift::a],
            V[ashift::a, ashift::a],
            units="xy",
            scale=l,
            scale_units="xy",
            width=w,
            angles="xy",
            pivot="mid",
            color=arrow_color,
        )

    if not color and a > 0:
        qk = ax.quiverkey(
            q, X=0.95, Y=0.98, U=1, label=r"$Msat$", labelpos="S", coordinates="axes"
        )
        qk.text.set_backgroundcolor("w")
        if origin == "upper":
            ax.invert_yaxis()

    if title is not None:
        tr = False
        ax.set_title(title)
    else:
        tr = True

    plt.tick_params(axis="x", labelbottom=False, bottom=False, top=False)
    plt.tick_params(axis="y", labelleft=False, left=False, right=False)
    # ax.set_aspect(aspect)
    plt.show()

    if save is not None:
        if not color:
            tr = False
        fig.set_size_inches(8, 8 / aspect)
        print(f"Saving: {save}")
        plt.axis("off")
        dpi = kwargs.get("dpi", max(dimy, dimx) * 5 / sz_inches)
        # sets dpi to 5 times original image dpi so arrows are reasonably sharp
        plt.savefig(save, dpi=dpi, bbox_inches="tight", transparent=tr)

    return


def show_fft(fft, title=None, **kwargs):
    """Display the log of the abs of a FFT

    Args:
        fft (ndarray): 2D image
        title (str, optional): title of image. Defaults to None.
        **kwargs: passed to show_im()
    """
    fft = np.copy(fft)
    nonzeros = np.nonzero(fft)
    fft[nonzeros] = np.log10(np.abs(fft[nonzeros]))
    fft = fft.real
    show_im(fft, title=title, **kwargs)


def show_log(im, title=None, **kwargs):
    """Display the log of an image

    Args:
        im (ndarray): 2D image
        title (str, optional): title of image. Defaults to None.
        **kwargs: passed to show_im()
    """
    im = np.copy(im)
    nonzeros = np.nonzero(im)
    im[nonzeros] = np.log(np.abs(im[nonzeros]))
    show_im(im, title=title, **kwargs)


def show_im_peaks(im=None, peaks=None, peaks2=None, size=None, title=None, **kwargs):
    """
    peaks an array [[y1,x1], [y2,x2], ...]
    """
    _fig, ax = plt.subplots()
    if im is not None:
        ax.matshow(im, cmap="gray", **kwargs)
    if peaks is not None:
        peaks = np.array(peaks)
        ax.plot(
            peaks[:, 1],
            peaks[:, 0],
            c="r",
            alpha=0.9,
            ms=size,
            marker="o",
            fillstyle="none",
            linestyle="none",
        )
    if peaks2 is not None:
        peaks2 = np.array(peaks2)
        ax.plot(
            peaks2[:, 1],
            peaks2[:, 0],
            c="b",
            alpha=0.9,
            ms=size,
            marker="o",
            fillstyle="none",
            linestyle="none",
        )
    ax.set_aspect(1)
    if title is not None:
        ax.set_title(str(title), pad=0)
    plt.show()


###
### Misc small helpers
###


def get_histo(im, minn=None, maxx=None, numbins=None):
    """
    gets a histogram of a list of datapoints (im), specify minimum value, maximum value,
    and number of bins
    """
    im = np.array(im)
    if minn is None:
        minn = np.min(im)
    if maxx is None:
        maxx = np.max(im)
    if numbins is None:
        numbins = min(np.size(im) // 20, 100)
        print(f"{numbins} bins")
    _fig, ax = plt.subplots()
    ax.hist(im, bins=np.linspace(minn, maxx, numbins))
    plt.show()


def get_fft(im):
    """Get fast fourier transform of 2D image"""
    return np.fft.fftshift(np.fft.fft2(im))


def get_ifft(fft):
    """Get inverse fast fourier transform of 2D image"""
    return np.fft.ifft2(np.fft.ifftshift(fft))


def Tukey2D(shape, alpha=0.5, sym=True):
    """
    makes a 2D (rectangular not round) window based on a Tukey signal
    Useful for windowing images before taking FFTs
    """
    dimy, dimx = shape
    ty = tukey(dimy, alpha=alpha, sym=sym)
    filt_y = np.tile(ty.reshape(dimy, 1), (1, dimx))
    tx = tukey(dimx, alpha=alpha, sym=sym)
    filt_x = np.tile(tx, (dimy, 1))
    output = filt_x * filt_y
    return output


def norm_image(image):
    """Normalize image intensities to between 0 and 1"""
    image = image - np.min(image)
    image = image / np.max(image)
    return image


def overwrite_rename(filepath):
    """Given a filepath, check if file exists already. If so, add numeral 1 to end,
    if already ends with a numeral increment by 1.

    Args:
        filepath (str): filepath to be checked

    Returns:
        str: [description]
    """

    filepath = str(filepath)
    file, ext = os.path.splitext(filepath)
    if os.path.isfile(filepath):
        if file[-1].isnumeric():
            file, num = splitnum(file)
            nname = file + str(int(num) + 1) + ext
            return overwrite_rename(nname)
        else:
            return overwrite_rename(file + "1" + ext)
    else:
        return filepath


def splitnum(s):
    """split the trailing number off a string. Returns (stripped_string, number)"""
    head = s.rstrip("-.0123456789")
    tail = s[len(head) :]
    return head, tail


###
### Pretty much everything else...
###


def autocorr(arr):
    """Calculate the autocorrelation of an image
    method described in from Loudon & Midgley, Ultramicroscopy 109, (2009).

    Args:
        arr (ndarray): Image to autocorrelate

    Returns:
        ndarray: Autocorrelation
    """
    fft = get_fft(arr)
    return np.fft.ifftshift(get_ifft(fft * np.conjugate(fft))).real


def bbox(img, digits=10):
    """
    Get minimum bounding box of image, trimming off black (0) regions.
    values will be rounded to `digits` places.
    """
    img = np.array(img)
    if img.dtype == "bool":
        img = img.astype("int")
    img = np.round(img, digits)
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return img[ymin : ymax + 1, xmin : xmax + 1]


def get_largest_rect(mask, skip=0, jit=True, precompile=True, v=1):
    """Get the largest rectangle in a binary mask image that does not contain a masked
    value.

    Args:
        mask (ndarray): 2D binary image
        skip (int, optional): "Bad" value. Defaults to 0.
        jit (bool, optional): Use JIT compilation to speed things up. Defaults to True.
        precompile (bool, optional): precompile the function. Defaults to True.

    Returns:
        ndarray: [[y1,x1], [y2,x2]] corner points of rectangle
    """
    vprint = print if v >= 1 else lambda *a, **k: None
    mask = np.array(mask).astype("int")
    stime = time.time()
    if jit:
        if precompile:
            _ = _get_largest_rect(mask[:10, :10], skip)
            stime = time.time()
        area_max = _get_largest_rect(mask, skip, v=1)
    else:
        nrows, ncols = mask.shape
        w = np.zeros_like(mask, dtype=int)
        h = np.zeros_like(mask, dtype=int)
        area_max = (0, [])
        for r in range(nrows):
            if r % 50 == 0:
                vprint(f"\rLargest rect: {r}/{nrows}", end="")
            for c in range(ncols):
                if mask[r, c] == skip:
                    continue
                if r == 0:
                    h[r, c] = 1
                else:
                    h[r, c] = h[r - 1, c] + 1
                if c == 0:
                    w[r, c] = 1
                else:
                    w[r, c] = w[r, c - 1] + 1
                minw = w[r, c]
                for dh in range(h[r, c]):
                    minw = min(minw, w[r - dh, c])
                    area = (dh + 1) * minw
                    if area > area_max[0]:
                        area_max = (area, [(r - dh, c - minw + 1, r, c)])
    vprint(f"Found largest rectangle in: {time.time() - stime:.1f}s")
    return np.array([area_max[1][:2], area_max[1][2:]])


@jit(nopython=True, parallel=False)
def _get_largest_rect(mask, skip, v=0):
    nrows, ncols = mask.shape
    w = np.zeros(mask.shape, dtype="int")
    h = np.zeros(mask.shape, dtype="int")
    area_max = (0, [0, 0, 0, 0])

    for r in range(nrows):
        for c in range(ncols):
            if mask[r, c] == skip:
                continue
            if r == 0:
                h[r, c] = 1
            else:
                h[r, c] = h[r - 1, c] + 1
            if c == 0:
                w[r, c] = 1
            else:
                w[r, c] = w[r, c - 1] + 1
            minw = w[r, c]
            for dh in numba.prange(h[r, c]):
                minw = min(minw, w[r - dh, c])
                area = (dh + 1) * minw
                if area > area_max[0]:
                    area_max = (area, [r - dh, c - minw + 1, r, c])

    return area_max


def filter_hotpix(
    image, thresh=12, show=False, iters=0, maxiters=10, ksize=3, thorough=False
):
    """
    look for pixel values with an intensity >3 std outside of mean of surrounding
    8 pixels. If found, replace with median value of those pixels

    thorough (bool): If false, will do default (fast) way of setting hot/dead pixels to
    the average of their neighbors. If True, will look at those neighbors and eliminate
    outliers before calculating average.
    """
    if iters > maxiters:
        print(f"Ended at {maxiters} iterations of filter_hotpix.")
        return image

    if int(ksize) % 2 != 1:
        ksize = int(ksize) + 1

    kernel = np.ones((ksize, ksize))
    kernel[ksize // 2, ksize // 2] = 0
    dimy, dimx = image.shape

    kernel = kernel / np.sum(kernel)
    image = image.astype("float")
    mean = ndi.convolve(image, kernel, mode="reflect")
    dif = np.abs(image - mean)
    std = np.std(dif)

    bads = np.where(dif > thresh * std)
    numbads = len(bads[0])

    filtered = np.copy(image)
    if thorough:
        # w = np.ones((3, 3))
        # w[1, 1] = 0
        # w = w.astype("bool")
        s = 5
        s1 = s2 = s3 = s4 = s
        for y, x in zip(bads[0], bads[1]):
            if y - s < 0:  # there's a better way to do this for sure.
                s1 = y
            if y + s + 1 > dimy:
                s2 = dimy - y - 1
            if x - s < 0:
                s3 = x
            if x + s + 1 > dimx:
                s4 = dimx - x - 1
            arr = np.copy(image[y - s1 : y + s2 + 1, x - s3 : x + s4 + 1])
            w = np.ones_like(arr).astype("bool")
            cy, cx = s1, s3
            w[cy, cx] = False
            mean2 = np.mean(arr, where=w)
            std2 = np.std(arr, where=w)
            arr[cy, cx] = np.nan
            keepsthresh = 2
            keeps = np.where(
                (mean2 - keepsthresh * std2 < arr) & (arr < mean2 + keepsthresh * std2)
            )
            if keeps[0].size == 0:
                print("no keeps for y,x, arrshape: ", y, x, arr.shape)
                nval = mean2
            else:
                nval = np.mean(arr[keeps])
            filtered[y, x] = nval
        pass
    else:
        # default
        filtered[bads] = mean[bads]
    if show:
        print(numbads, "hot-pixels filtered")
        show_im_peaks(
            image,
            np.transpose([bads[0], bads[1]]),
            title="hotpix identified on first pass",
        )
        show_im(filtered, "filtered image")
    if numbads > 0:
        filtered = filter_hotpix(
            filtered,
            thresh=thresh,
            show=False,
            iters=iters + 1,
            maxiters=maxiters,
            ksize=ksize,
            thorough=thorough,
        )

    return filtered


def filter_background(
    image,
    scale=1,
    filt_hotpix=True,
    thresh=15,
    #filter_lf=100,
    #filter_hf=10,
    filter_lf=100,
    filter_hf=10,
    show=False,
    ret_bkg=False,
    thorough=True,
):
    """
    image: image to be filtered
    scale: scale of image in nm/pixel, this allows you to set the filter sizes in nm
    filt_hotpix: True if you want to filter hot/dead pixels, false otherwise
    thresh: threshold for hotpix filtering. Higher threshold means fewer pixels
        will be filtered
    filter_lf: low-frequency filter std in nm (or pix if no scale)
    filter_hf: high-frequeuency filter std in nm (or pix if no scale)
    ret_bkg: will return the subtracted background (no hotpix) if True

    returns filtered_im if ret_bkg False
    returns (filtered_im, background) if ret_bkg True
    """
    dim_y, dim_x = image.shape

    x_sampling = y_sampling = 1 / scale  # [pixels/nm]
    u_max = x_sampling / 2
    v_max = y_sampling / 2
    u_axis_vec = np.linspace(-u_max / 2, u_max / 2, dim_x)
    v_axis_vec = np.linspace(-v_max / 2, v_max / 2, dim_y)
    u_mat, v_mat = np.meshgrid(u_axis_vec, v_axis_vec)
    r = np.sqrt(u_mat ** 2 + v_mat ** 2)  # nm

    if filter_lf is not None:
        inverse_gauss_filter = 1 - np.exp(-1 * (r * filter_lf) ** 2)
    else:
        inverse_gauss_filter = np.ones_like(r)
    if filter_hf is not None:
        gauss_filter = np.exp(-1 * (r * filter_hf) ** 2)
    else:
        gauss_filter = np.ones_like(r)
    bp_filter = inverse_gauss_filter * gauss_filter

    orig_im = np.copy(image)
    if filt_hotpix:
        image = filter_hotpix(image, show=show, thresh=thresh, thorough=thorough)
        if show:
            show_im(image, "image after hotpix filter")
    fft = get_fft(image)

    ffilt = not np.all(bp_filter == 1)
    if ffilt:
        filtered_im = np.real(get_ifft(fft * bp_filter))
        dif = image - filtered_im
    else:
        filtered_im = image
        dif = orig_im - filtered_im

    if show:
        if ffilt:
            show_im(bp_filter, "filter", simple=True)
        show_im(filtered_im, "filtered image", cbar=False, scale=scale)
        show_im(dif, "removed background", simple=True)
        # show_fft(get_fft(dif), 'fft of background', cbar=False)

    filtered_im = norm_image(filtered_im)
    if ret_bkg:
        return (filtered_im, norm_image(dif))
    else:
        return filtered_im


def dist(ny, nx, shift=False):
    """Creates a frequency array for Fourier processing.

    Args:
        ny (int): Height of array
        nx (int): Width of array
        shift (bool): Whether to center the frequency spectrum.

            - False: (default) smallest values are at the corners.
            - True: smallest values at center of array.

    Returns:
        ``ndarray``: Numpy array of shape (ny, nx).
    """
    ly = (np.arange(ny) - ny / 2) / ny
    lx = (np.arange(nx) - nx / 2) / nx
    [X, Y] = np.meshgrid(lx, ly)
    q = np.sqrt(X ** 2 + Y ** 2)
    if not shift:
        q = np.fft.ifftshift(q)
    return q


def dist4(dim, norm=False):
    """4-fold symmetric distance map even at small radiuses

    Args:
        dim (int): desired dimension of output
        norm (bool, optional): Normalize maximum of output to 1. Defaults to False.

    Returns:
        ``ndarray``: 2D (dim, dim) array
    """
    # 4-fold symmetric distance map even at small radiuses
    d2 = dim // 2
    a = np.arange(d2)
    b = np.arange(d2)
    if norm:
        a = a / (2 * d2)
        b = b / (2 * d2)
    x, y = np.meshgrid(a, b)
    quarter = np.sqrt(x ** 2 + y ** 2)
    sym_dist = np.zeros((dim, dim))
    sym_dist[d2:, d2:] = quarter
    sym_dist[d2:, :d2] = np.fliplr(quarter)
    sym_dist[:d2, d2:] = np.flipud(quarter)
    sym_dist[:d2, :d2] = np.flipud(np.fliplr(quarter))
    return sym_dist


def circ4(dim, rad=None):
    """Binary circle mask, 4-fold symmetric even at small dimensions"""
    if rad is None:
        rad = dim // 2
    return (dist4(dim) < rad).astype("int")


def lineplot_im(
    image,
    center=None,
    phi=0,
    linewidth=1,
    line_len=-1,
    show=False,
    use_abs=False,
    save=None,
    **kwargs,
):
    """
    image to take line plot through
    center point (cy, cx) in pixels
    angle (deg) to take line plot, with respect to y,x axis (y points down).
        currently always does the line profile left to right,
        phi=90 will be vertical profile top to bottom
        phi = -90 will be vertical profile bottom to top
    line_len (pix) length of line scan
    """
    im = np.array(image)
    if np.ndim(im) > 2:
        print("More than 2 dimensions given, collapsing along first axis")
        im = np.sum(im, axis=0)
    dy, dx = im.shape
    if center is None:
        print("line through middle of image")
        center = (dy // 2, dx // 2)
    cy, cx = center[0], center[1]

    sp, ep = box_intercepts(im.shape, center, phi, line_len)

    profile = skimage.measure.profile_line(
        im, sp, ep, linewidth=linewidth, mode="constant", reduce_func=np.mean
    )
    if line_len > 0 and len(profile) > line_len:
        lp = int(len(profile))
        profile = profile[(lp - line_len) // 2 : -(lp - line_len) // 2]

    if show:
        show_scan = kwargs.get("show_scan", True)
        if show_scan:
            _fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
            if use_abs:
                ax0.plot(np.abs(profile))
            else:
                ax0.plot(profile)
            ax0.set_aspect(1 / ax0.get_data_ratio(), adjustable="box")
            ax0.set_ylabel("intensity")
            ax0.set_xlabel("pixels")
        else:
            _fig, ax1 = plt.subplots()

        cmap = kwargs.get("cmap", "gray")
        img = ax1.matshow(
            im, cmap=cmap, vmin=kwargs.get("vmin", None), vmax=kwargs.get("vmax", None)
        )
        if kwargs.get("cbar", False):
            plt.colorbar(img, ax=ax1, pad=0.02)

        if linewidth > 1:
            th = np.arctan2((ep[0] - sp[0]), (ep[1] - sp[1]))
            spp, epp = box_intercepts(
                im.shape,
                (cy + np.cos(th) * linewidth / 2, cx - np.sin(th) * linewidth / 2),
                phi,
                line_len,
            )
            spm, epm = box_intercepts(
                im.shape,
                (cy - np.cos(th) * linewidth / 2, cx + np.sin(th) * linewidth / 2),
                phi,
                line_len,
            )
            color = kwargs.get("color", "r")
            # ax1.plot([spp[1], epp[1]], [spp[0], epp[0]], color=color, linewidth=1)
            # ax1.plot([spm[1], epm[1]], [spm[0], epm[0]], color=color, linewidth=1)
            ax1.fill(
                [spp[1], epp[1], epm[1], spm[1]],
                [spp[0], epp[0], epm[0], spm[0]],
                alpha=0.3,
                facecolor=color,
                edgecolor=None,
            )
            ax1.plot([sp[1], ep[1]], [sp[0], ep[0]], color=color, linewidth=0.5)

        else:
            ax1.plot(
                [sp[1], ep[1]],
                [sp[0], ep[0]],
                color=kwargs.get("color", "r"),
                linewidth=1,
            )

        ax1.set_xlim([0, im.shape[1] - 1])
        ax1.set_ylim([im.shape[0] - 1, 0])

        if save:
            print("saving: ", save)
            plt.savefig(save, dpi=kwargs.get("dpi", 600), bbox_inches="tight")

        plt.show()

    return profile


def box_intercepts(dims, center, phi, line_len=-1):
    """
    given box of size dims=(dy, dx), a line at angle phi (deg) with respect to the x
    axis and going through point center=(cy,cx) will intercept the box at points
    sp = (spy, spx) and ep=(epy,epx) where sp is on the left half and ep on the
    right half of the box. for phi=90deg vs -90 will flip top/bottom sp ep
    """
    dy, dx = dims
    cy, cx = center
    phir = np.deg2rad(phi)
    tphi = np.tan(phir)
    tphi2 = np.tan(phir - np.pi / 2)

    # calculate the end edge
    epy = round((dx - cx) * tphi + cy)
    if 0 <= epy < dy:
        epx = dx - 1
    elif epy < 0:
        epy = 0
        epx = round(cx + cy * tphi2)
    else:
        epy = dy - 1
        epx = round(cx + (dy - cy) / tphi)

    spy = round(cy - cx * tphi)
    if 0 <= spy < dy:
        spx = 0
    elif spy >= dy:
        spy = dy - 1
        spx = round(cx - (dy - cy) * tphi2)
    else:
        spy = 0
        spx = round(cx - cy / tphi)

    if line_len > 0:
        sp2y = cy - np.sin(np.deg2rad(phi)) * line_len / 2
        sp2x = cx - np.cos(np.deg2rad(phi)) * line_len / 2
        ep2y = cy + np.sin(np.deg2rad(phi)) * line_len / 2
        ep2x = cx + np.cos(np.deg2rad(phi)) * line_len / 2
        spy = spy if sp2y < 0 else sp2y
        spx = spx if sp2x < 0 else sp2x
        epy = epy if ep2y > dy - 1 else ep2y
        epx = epx if ep2x > dx - 1 else ep2x

    sp = (spy, spx)  # start point
    ep = (epy, epx)  # end point
    return sp, ep


def total_tilt(tx, ty, xfirst=True, rad=False):
    """
    returns (altitude, azimuth) in degrees after tilting around x axis by tx
    and then y axis by ty.
    xfirst=True if rotating around x then y, affects azimuth only
    rad=False if input degrees (default) or True if input in radians
    """
    if not rad:
        tx = np.deg2rad(tx)
        ty = np.deg2rad(ty)
    Rx = R.from_rotvec(tx * np.array([1, 0, 0]))  # [x,y,z]
    Ry = R.from_rotvec(ty * np.array([0, 1, 0]))
    v = np.array([0, 0, 1])
    if xfirst:
        vrot = Ry.apply(Rx.apply(v))
    else:
        vrot = Rx.apply(Ry.apply(v))

    alt = np.arctan(np.sqrt(vrot[0] ** 2 + vrot[1] ** 2) / vrot[2])
    alt = round(np.rad2deg(alt), 13)

    az = np.rad2deg(np.arctan2(vrot[1], vrot[0]))
    return alt, az


###
### Helpers that are more specific and used by other functions I've written
### These should (and likely will) be moved out of this document at some point
###


def get_mean(pos1, pos2):
    """Mean point of two positions (2D)"""
    return ((pos1[0] + pos2[0]) / 2, (pos1[1] + pos2[1]) / 2)


def get_dist(pos1, pos2):
    """Distance between two 2D points"""
    squared = (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2
    return np.sqrt(squared)


def sort_clockwise(points, ref):
    """Sort a set of points into a clockwise order"""
    points = np.array(points)
    points = points.astype("float")
    points = np.unique(points, axis=0)
    angles = np.arctan2(points[:, 0] - ref[0], points[:, 1] - ref[1])
    ind = np.argsort(angles)
    return points[ind[::-1]]


def center_crop_square(im, ret_corners=False):
    """Crop image to square using min(dimy, dimx)
    if ret_corners will return: ((y1, x1), (y2,x2)) ((upper left), (lower right))
    """
    dy, dx = im.shape
    if dy == dx:
        if ret_corners:
            return ((0, 0), (dy, dx))
        else:
            return im
    elif dy > dx:
        my = dy // 2
        dx1 = int(np.ceil(dx / 2))
        dx2 = int(np.floor(dx / 2))
        y1 = my - dx1
        y2 = my + dx2
        if ret_corners:
            return ((y1, 0), (y2, dx))
        else:
            return im[y1:y2, :]
    elif dy < dx:
        mx = dx // 2
        dy1 = int(np.ceil(dy / 2))
        dy2 = int(np.floor(dy / 2))
        x1 = mx - dy1
        x2 = mx + dy2
        if ret_corners:
            return ((0, x1), (dy, x2))
        else:
            return im[:, x1:x2]


def center_crop_im(image, shape, dim_order_in="channels_last"):
    """crop image to (shape) keeping the center of the image

    Args:
        image (_type_): _description_
        shape (_type_): _description_
        dim_order_in (str, optional): _description_. Defaults to "channels_last".

    Returns:
        _type_: _description_
    """
    if image.ndim == 2:
        dimy, dimx = image.shape
    elif image.ndim == 3:
        if dim_order_in == "channels_last":
            dimy, dimx, dimz = image.shape
        elif dim_order_in == "channels_first":
            dimz, dimy, dimx = image.shape

    dyf, dxf = shape

    if dyf > dimy:
        cy1 = int(np.floor((dyf - dimy) / 2))
        cy2 = int(np.ceil((dyf - dimy) / 2))
        image = np.pad(image, ((cy1, cy2), (0, 0)))
        dimy = dyf
    if dxf > dimx:
        cx1 = int(np.floor((dxf - dimx) / 2))
        cx2 = int(np.ceil((dxf - dimx) / 2))
        image = np.pad(image, ((0, 0), (cx1, cx2)))
        dimx = dxf

    if dimy > dyf:
        cropt = int(np.floor((dimy - dyf) / 2))
        cropb = -1 * int(np.ceil((dimy - dyf) / 2))
    else:
        cropt, cropb = 0, dimy
    if dimx > dxf:
        cropl = int(np.floor((dimx - dxf) / 2))
        cropr = -1 * int(np.ceil((dimx - dxf) / 2))
    else:
        cropl, cropr = 0, dimx

    if dim_order_in == "channels_last":
        return image[cropt:cropb, cropl:cropr]
    elif dim_order_in == "channels_first":
        return image[:, cropt:cropb, cropl:cropr]
