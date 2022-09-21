import numpy as np
import matplotlib.pyplot as plt
from image_helpers import show_im, show_im_peaks, circ4, sort_clockwise, get_dist
from scipy.ndimage import center_of_mass
from skimage.feature import peak_local_max
from pathlib import Path
from skimage import io


class DP_Im(object):
    """
    future things maybe relevant but not implementing for now:
        - size of peaks
        - dealing with a beam block

    Attributes:
        im : raw image data np array
        scale : scale of image (1/nm)
        threshold : threshold value used to find peaks
        peakmaxrad : radius used for summing area and getting peak intensity

        peaks : array of peak locations and intensities, [[y1, x1, intensity1], [y2, x2...], ...]
            peaks[0] should always be the central peak
    """

    def __init__(self, filepath=None, image=None, name="", scale=None):
        self.name = name
        self.scale = scale
        self.peaks = None
        self.threshold = None
        self.filepath = filepath

        if filepath is not None:
            if isinstance(filepath, str):
                p = Path(filepath)
                self.filepath = p

        if image is not None:  # image data given directly
            self.im = np.array(image).astype("float")
            self.shape = np.shape(image)
            if self.scale is None:
                print("No scale given.")
        else:  # filepath given instead
            self.load_im()

    @classmethod
    def load(cls, filepath=None, scale_factor=1, scale=None, imdata=None):
        """
        load and parse title
        """
        pass
        # Im = DP_im(
        #     filepath=filepath,
        #     image_name=Tilt_name,
        #     tilt_angle=tilt_angle,
        #     imdata=imdata,
        # )

    def load_im(self):
        """
        load an image
        """
        p = self.filepath
        if p.suffix in [".dm3", ".dm4"]:
            self.load_dm()
        elif p.suffix in [".tiff", ".tif", ".png", ".bmp", ".gif"]:
            # use skimage.io rather than hs to try and reduce the hs dependency
            im = io.imread(p)
            self.im = im.astype("float")
            self.shape = im.shape
            if self.scale is None:
                print("No scale loaded from image file")
        return

    def load_dm(self):
        import hyperspy.api as hs  # big import and unfortunate dependency

        s2d = hs.load(self.filepath)
        s2d.axes_manager.convert_units(units=["1/nm", "1/nm"])  # nm/pix
        assert s2d.axes_manager[0].scale == s2d.axes_manager[1].scale
        self.scale = s2d.axes_manager[0].scale
        self.axes_manager = s2d.axes_manager
        self.metadata = s2d.metadata
        self.im = s2d.data.astype("float")
        self.shape = s2d.data.shape
        s2d = None
        return

    def find_peaks(self, threshold_rel=0.005, min_distance=50, show=False, mrad=20):
        """Identify peaks in the diffraction pattern using local maxima then refining with center of mass
        over windowed region

        Args:
            threshold_rel (float, optional): Peak threshold value (0,1). Defaults to 0.005.
            min_distance (int, optional): Minimum distance between found peaks (pixels). Defaults to 50.
            show (bool, optional): Show found peaks. Defaults to False.
            mrad (int, optional): Radius of window used. Defaults to 20.
        """
        local_maxes = peak_local_max(
            self.im, min_distance=min_distance, threshold_rel=threshold_rel
        )

        # use masked radius and get center of mass
        peaks = []
        mask = circ4(2 * mrad)
        for y1, x1 in local_maxes:  # should handle edge cases
            subregion = self.im[y1 - mrad : y1 + mrad, x1 - mrad : x1 + mrad]
            ny1, nx1 = center_of_mass(subregion * mask)
            intensity = np.sum(subregion * mask)
            peaks.append([y1 - mrad + ny1, x1 - mrad + nx1, intensity])

        if show:
            im = np.copy(self.im)
            nonzeros = np.nonzero(im)
            im[nonzeros] = np.log(np.abs(im[nonzeros]))
            show_im_peaks(im, peaks)

        peaks = np.array(peaks)
        peaks_sort = peaks[np.argsort(peaks[:, 2])[::-1]]
        self.peaks = peaks_sort

    def show_DP_peaks(self, title=None, markersize=None, save=False, **kwargs):
        if self.peaks is None:
            print("no peaks found")
            return

        im = np.copy(self.im)

        if title is None:
            fig = plt.figure()
            aspect = im.shape[0] / im.shape[1]
            figsize = kwargs.get("figfigsize", (4, 4 * aspect))
            if isinstance(figsize, (int, float)):
                figsize = (figsize, figsize)
            fig.set_size_inches(figsize)
            ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
            fig.add_axes(ax)
        else:
            _fig, ax = plt.subplots()

        nonzeros = np.nonzero(im)
        im[nonzeros] = np.log(np.abs(im[nonzeros]))
        ax.matshow(im, cmap="gray", **kwargs)
        ax.plot(
            self.peaks[:, 1],
            self.peaks[:, 0],
            c="r",
            alpha=0.9,
            ms=markersize,
            marker="o",
            fillstyle="none",
            linestyle="none",
        )

        for i, (y, x, intensity) in enumerate(self.peaks):
            plt.text(x + 10, y - 10, f"{i}", color="white", ha="left", va="bottom")

        if title is not None:
            ax.set_title(str(title), pad=0)

        ax.set_axis_off()
        if save:
            print("saving: ", save)
            dpi = kwargs.get("dpi", 400)
            plt.savefig(save, dpi=dpi, bbox_inches=0)
        plt.show()


def get_peak_dists(DP, inds):
    cent = DP.peaks[0, :2]
    dists = []
    for peak in DP.peaks[inds][:, :2]:
        dists.append(get_dist(cent, peak))

    dists = np.array(dists)
    dists *= DP.scale
    dists = 1 / dists
    # conver to 1/nm
    return dists


def get_peak_angles(DP, inds):
    cent = DP.peaks[0, :2]
    # peaks = [DP.peaks[i][:2] for i in inds]
    # peaks = sort_clockwise(peaks)

    angles = []
    for i in range(len(inds)):
        j = i + 1 if i + 1 < len(inds) else 0
        p1 = DP.peaks[inds[i], :2]
        p2 = DP.peaks[inds[j], :2]
        angles.append(get_angle(p1, cent, p2))
    return np.array(angles)


def get_angle(a, b, c):
    """angle formed by a -> b -> c, positive is counterclockwise
    a = [y1, x1], b = [y2, x2], c = [y3, x3]"""
    ang = np.rad2deg(
        np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    )
    return ang + 360 if ang <= -180 else ang
