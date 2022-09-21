import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import dill
import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import skimage
from skimage import io
from Skyrm import Skyrm

from image_helpers import (
    bbox,
    center_crop_square,
    filter_hotpix,
    get_fft,
    get_ifft,
    norm_image,
    overwrite_rename,
    show_fft,
    show_im,
    splitnum,
    total_tilt,
)


class Tilt_Im(object):
    """object for each image


    Attributes:
        orig_im (2D array): raw image data loaded from the .dm3
        full_im (2D array): raw image that has been tilt corrected
        im (2D array): filtered and cropped region of full_im. This image is
            what is used for all analysis.
        skyrm_list (list): List of Skyrm objects
        axes_manager (thing): hyperspy signal2D axes_manager object
        scale (float): scale of image nm/pix
        shape ([y,x]): Shape of im (not full image)
        metadata (thin): dm3 metadata

        crop : cropped region boundaries


        filepath (str): Path to original dm3 image
        name (str): Name given to image and Tilt_Im object

        fft_skyrm_size (2D array): [size(nm), std(nm)] # nm
        rescale (float): rescaling factor
        mbox (ndarray): [[y1,x1], [y2,x2]] corner points of box containing only
            skyrmions and no disruptions/background/regions skyrms could not be found.
            Useful when calculating RDFs or anything where uniform skyrms are needed.
        modified (datetime): datetime of when Tilt_im was last saved


    skyrm_aves:
        "psi6abs" = [np.mean(psi6abs_lst), np.std(psi6abs_lst)]
        "psi4abs" = [np.mean(psi4abs_lst), np.std(psi4abs_lst)]
        "psi6cabs" = [np.mean(psi6cabs_lst), np.std(psi6cabs_lst)]
        "psi4cabs" = [np.mean(psi4cabs_lst), np.std(psi4cabs_lst)]
        "num_neighbors" = [np.mean(num_neighbors_lst),
                                              np.std(num_neighbors_lst)]
        "vor_area" = [np.mean(vor_area_lst),
                                         np.std(vor_area_lst)] # pixels^2
        "vor_circ_rad" = [np.mean(vor_circ_rad_lst),
                                             np.std(vor_circ_rad_lst)] # pixels
        'im_scale' = scale # nm/pix
        'num_skyrms' = [num_not_edge, len(skyrm_list)]
        'im_dims' = shape # pix
        'fft_skyrm_size' = fft_skyrm_size # diameter, nm


    """

    def __init__(self, filepath=None, name="", tilt_angle=0, imdata=None):
        self.skyrm_list = []
        self.filepath = filepath
        self.name = name
        self.scale = None  # nm/pixel of image
        self.rescale = None  # scaling factor before applying NN
        self.axes_manager = None  # axes manager from DM3
        self.metadata = None
        self.orig_im = None  # original image as loaded from file
        self.im = None  # will be the filtered/cropped image actually used
        self.full_im = (
            None  # tilt corrected version of orig image. (maybe not necessary)
        )
        self.shape = None  # self.im.shape

        self.tilt_angle = tilt_angle  # degree of tilt, +/- affects tilt_dir (+/- 180)
        self.tilt_dir = (
            None  # direction along which the image is tilted, (put white on right)
        )
        self.fft_skyrm_size = (
            None  # average skyrmion diameter in nm from FFT (approximate)
        )
        self.crop = None  # dict of slices used to crop self.full_im -> self.im
        self.mbox = None  # box corners containing only skyrms
        self.save_path = None  # where the Tilt_Im object is saved

        self.rGr = None  # Radial distribution function [r, Gr]
        self.rGr6 = None  # Orientationa correlation function [r, Gr6] from psi6
        self.rGr6c = None  # Orientationa correlation function [r, Gr6] from psi6c
        self.skyrm_aves = None  # dict containing the skyrmion average values for
        # psi6abs, psi4abs, num_neighbors, vor_area, vor_circ_rad
        self.modified = None

        if imdata is not None:
            imdata = np.array(imdata)
            self.orig_im = imdata
            self.shape = imdata.shape

        elif self.filepath is not None:
            if filepath.suffix == ".dm3" or filepath.suffix == ".dm4":
                self.load_dm3()
            else:
                self.load_im()

    @classmethod
    def load(cls, filepath=None, scale_factor=1, scale=None, imdata=None):
        """
        takes a dm3, parses the title for data, and creates Tilt_Im object
        """

        filepath = Path(filepath)
        fname = filepath.stem
        mdata = fname.split("_")
        tx, ty, temp, field, mode = None, None, None, None, None
        for info in mdata:
            info = info.lower()
            if info.startswith("tx"):
                tx = float(splitnum(info)[1])
            if info.startswith("ty"):
                ty = float(splitnum(info)[1])
            if info.endswith("k") and not info.startswith("x"):
                if info.startswith("t"):
                    info = info[1:]
                temp = info[:-1]
            if info.endswith("g") or info.startswith("b"):  # might not work for all
                if info.endswith("g"):
                    info = info[:-1]
                field = splitnum(info)[1]

        fpath = Path(filepath)
        fullPath = str(fpath.parent).lower()
        if "cool" in fullPath or "fc" in fullPath:
            mode = "FC"
        elif "heat" in fullPath or "fh" in fullPath:
            mode = "FH"

        # calculate total tilt off of z axis
        if tx is not None or ty is not None:
            tilt_angle = round(total_tilt(tx, ty)[0], 1)

        Tilt_name = ""
        if mode is not None:
            Tilt_name += mode + "_"
        if field is not None:
            Tilt_name += "B" + field + "G_"
        if temp is not None:
            Tilt_name += temp + "K"
        if Tilt_name.endswith("_"):
            Tilt_name = Tilt_name[:-1]

        Im = Tilt_Im(
            filepath=filepath,
            name=Tilt_name,
            tilt_angle=tilt_angle,
            imdata=imdata,
        )

        dmfiles = [".dm3", ".dm4", ".dm5"]
        if scale is not None:
            Im.set_scale(scale * scale_factor)
        elif filepath.suffix in dmfiles:
            Im.set_scale(Im.scale * scale_factor)  # account for defocus z
        elif scale_factor != 1:
            print("Scale factor given but no scale found. Setting scale=scale_factor")
            Im.set_scale(scale_factor)
        else:
            print("No scale given or found. setting scale = 1")
            Im.set_scale(1)
        return Im

    def load_dm3(self):
        s2d = hs.load(self.filepath)
        s2d.axes_manager.convert_units(units=["nm", "nm"])  # nm/pix
        assert s2d.axes_manager[0].scale == s2d.axes_manager[1].scale
        self.scale = s2d.axes_manager[0].scale
        self.axes_manager = s2d.axes_manager
        self.metadata = s2d.metadata
        self.orig_im = s2d.data
        self.shape = s2d.data.shape
        s2d = None
        return

    def load_im(self):
        """
        called if image is not a tiff
        """
        # use skimage.io rather than hs to try and reduce the hs dependency
        im = io.imread(self.filepath)
        self.orig_im = im
        self.shape = im.shape
        # if self.scale is None:
        #     print("No scale loaded from image file")
        return

    def fix_tilt(self, bbox_digits=None):
        """
        Adjust for the sample tilt in the microscope.
        Needs self.tilt_dir and self.tilt_angle to be set first.
        Assumes a large depth of focus relative to the fov and a parallel
        beam such that perspective effects are minimized
        """
        if bbox_digits is None:  # apply default/pick values for bbox
            if np.max(self.orig_im) > 10:
                bbox_digits = 0
            else:
                bbox_digits = 3

        tilt_scale = 1 / np.cos(np.deg2rad(self.tilt_angle))
        rot1 = ndi.rotate(np.copy(self.orig_im), self.tilt_dir, reshape=True, cval=0)
        dim_y, dim_x = rot1.shape
        scaled_im = skimage.transform.resize(
            rot1,
            (dim_y, int(dim_x * tilt_scale)),
            mode="constant",
            cval=0,
            preserve_range=True,
        )
        rot2 = ndi.rotate(scaled_im, -1 * self.tilt_dir, reshape=True, cval=0)
        if bbox_digits >= 0:  # if bbox_digits is < 0, do not bbox
            trimmed = bbox(rot2, bbox_digits)
        else:
            trimmed = rot2
        self.full_im = trimmed
        if self.im is None:
            self.im = np.copy(trimmed)
        self.shape = trimmed.shape
        return

    def crop_im(self, crop=None, reset=False):
        if reset or crop is None:  # resets to full_im (which is tilt adjusted)
            reset = True
            if self.full_im is not None:
                dimy, dimx = self.full_im.shape
                self.im = self.full_im
            else:
                dimy, dimx = self.orig_im.shape
                self.im = self.orig_im
            crop = {"top": 0, "bottom": dimy, "left": 0, "right": dimx}
            self.shape = (dimy, dimx)
            self.crop = crop
            return

        if self.full_im is not None:
            cim = self.full_im
        elif self.im is not None:
            cim = self.im
        else:
            cim = self.orig_im

        im = np.copy(cim[crop["top"] : crop["bottom"], crop["left"] : crop["right"]])
        self.im = norm_image(im)
        self.shape = im.shape
        self.crop = crop

    def filter_background(
        self,
        filt_hotpix=True,
        show=False,
        ret_bkg=False,
        filter_lf=None,
        thresh=15,
        orig=False,
        filter_hf=None,
        thorough=True,
    ):
        """
        this calls "filter_background" from hipl image_helpers
        """

        if orig:
            image = self.orig_im
        else:
            if self.crop is not None:
                self.crop_im(
                    self.crop
                )  # do this in case the image has been filtered already
            # this will reset self.im to the cropped (raw) original image
            if self.im is not None:
                image = self.im
            elif self.full_im is not None:
                image = self.full_im
            else:
                image = self.orig_im

        if filter_lf is None:
            if self.fft_skyrm_size:
                filter_lf = (
                    self.fft_skyrm_size[0] * 6
                )  # cutoff size nm, *6 to remove only lower freq noise
            else:
                filter_lf = 1200  # nm

        from image_helpers import filter_background

        filtered_im, bkg = filter_background(
            image,
            scale=self.scale,
            filt_hotpix=filt_hotpix,
            thresh=thresh,
            filter_lf=filter_lf,
            filter_hf=filter_hf,
            show=show,
            ret_bkg=True,
            thorough=thorough,
        )

        # dim_y, dim_x = image.shape

        # x_sampling = y_sampling = 1 / self.scale  # [pixels/nm]
        # u_max = x_sampling / 2
        # v_max = y_sampling / 2
        # u_axis_vec = np.linspace(-u_max / 2, u_max / 2, dim_x)
        # v_axis_vec = np.linspace(-v_max / 2, v_max / 2, dim_y)
        # u_mat, v_mat = np.meshgrid(u_axis_vec, v_axis_vec)
        # r = np.sqrt(u_mat ** 2 + v_mat ** 2)

        # if filter_width is None:
        #     if self.fft_skyrm_size:
        #         filter_width = (
        #             self.fft_skyrm_size[0] * 6
        #         )  # cutoff size nm, *6 to remove only lower freq noise
        #     else:
        #         filter_width = 1200  # nm
        # inverse_gauss_filter = 1 - np.exp(-1 * (r * filter_width) ** 2)

        # if filter_width2 is not None:
        #     gauss_filter = np.exp(-1 * (r * filter_width2) ** 2)
        # else:
        #     gauss_filter = np.ones_like(inverse_gauss_filter)

        # if filt_hotpix:
        #     image = filter_hotpix(image, show=show, thresh=thresh)
        # fft = get_fft(image)

        # filtered_im = np.real(get_ifft(fft * inverse_gauss_filter * gauss_filter))
        # dif = image - filtered_im

        # if show:
        #     show_im(filtered_im, "filtered image", cbar=False)
        #     show_im(inverse_gauss_filter * gauss_filter, "filter")
        #     show_im(dif, "removed background", cbar=False)
        #     # show_fft(get_fft(dif), 'fft of background', cbar=False)

        self.im = filtered_im

        if ret_bkg:
            return bkg
        else:
            return

        # def get_tilt_axis1(self, show=False):
        """
        Gets the tilt orientation by looking at the FFT, and measure the total intensity
        on a line profile through the center as the FFT is rotated. The minimum intensity
        point is used

        needs to be rewritten completely
        """

    #     sig = min(4, self.shape[0]//100)
    #     fft = ndi.gaussian_filter(np.abs(get_fft(self.orig_im)), sigma=sig)

    #     sums = []
    #     minsum = np.inf
    #     minangle = 0
    #     angles = np.arange(0, 180, 1)

    #     qlength = fft.shape[1]//4
    #     hwidth = 2

    #     print("Getting tilt axis (method 1), this takes a minute.")
    #     for i in angles:
    #         rot = ndi.rotate(fft, i)
    #         dy, dx = np.shape(rot)
    #         dy2 = dy//2
    #         dx2 = dx//2
    #         linesum = np.sum(rot[dy2-hwidth:dy2+hwidth+1, dx2-qlength:dx2+qlength])
    #         if linesum < minsum:
    #             minsum=linesum
    #             minangle = i
    #         sums.append(linesum)

    #     indices = range(minangle-30,minangle+31)
    #     n_angles = np.array(angles).take(indices, mode='wrap')
    #     n_linesums = np.array(sums).take(indices, mode='wrap')

    #     i = 0
    #     first = n_angles[0]
    #     for ang in n_angles:
    #         if ang < first:
    #             n_angles[i] += 180
    #         i+=1

    #     p = np.polyfit(n_angles, n_linesums, deg=2)
    #     fit = np.poly1d(p)
    #     xp = np.linspace(np.min(n_angles), np.max(n_angles), 500)
    #     minind = np.where(fit(xp) == np.min(fit(xp)))
    #     min_angle = xp[minind][0] # this is the direction of rotation, want axis

    #     ## display stuff
    #     if show:
    #         fig, ax = plt.subplots()
    #         ax.plot(n_angles, n_linesums)
    #         ax.plot(xp, fit(xp))
    # #         plt.axvspan(minangle, minangle+0.05, edgecolor='black')
    #         plt.axvspan(min_angle, min_angle+0.05, edgecolor='red')
    #         plt.show()

    #     if min_angle >=180:
    #         self.tilt_axis = min_angle-180
    #     elif min_angle<90 and min_angle>0:
    #         self.tilt_axis = min_angle+90
    #     else:
    #         print('You find yourself in a beautiful houes, with a beautiful wife')
    # return

    def get_tilt_dir(
        self,
        min_sigma=15,
        max_sigma=30,
        threshold=1200,
        show=False,
        pick=False,
        bbox_im=True,
        fix_tilt=True,
        bbox_digits=-1,
        v=1,
    ):
        """
        tilt_dir is direction along which image is tilted (with white set to the
        right, fixed by +/- tilt angle)
        """
        vprint = print if v >= 1 else lambda *a, **k: None

        if self.shape[0] == self.shape[1]:
            # if cropped_im and self.im is not None:
            #     fft = np.abs(get_fft(self.im))
            # else:
            # TODO: figure out if
            # this should be repeatable, should always use the orig_im?
            fft = np.abs(get_fft(self.orig_im))
        else:  # crop to square so fft scales are equal in both directions
            square_im = center_crop_square(self.orig_im)
            fft = np.abs(get_fft(square_im))

        blobs = skimage.feature.blob_dog(
            -1 * fft, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold
        )

        if len(blobs) > 2:
            vprint(
                f"""{len(blobs)} blobs found, this might be a problem.\nYou should probably increase the threshold."""
            )
        elif len(blobs) < 2:
            vprint(
                f"""{len(blobs)} blobs found.\nYou should probably decrease the threshold."""
            )
            if show:
                show_fft(fft)
            return "No blobs found"

        dy2, dx2 = np.array(fft.shape) // 2

        if pick:
            vprint("points: ")
            for i, blob in enumerate(blobs):
                deg = np.rad2deg(np.arctan2(blob[0] - dy2, blob[1] - dx2))
                vprint(f"ind: {i}, pos: {blob[:2]}, deg: {deg:.1f}")

            ind1 = int(input("Choose point 1 (int): "))
            ind2 = int(input("Choose point 2 (int): "))
            b1 = blobs[ind1]
            b2 = blobs[ind2]
        else:
            b1 = blobs[0]
            b2 = blobs[-1]

        if show:
            _fig, ax = plt.subplots()
            ax.imshow(np.log10(fft), cmap="gray")
            for blob in blobs:
                if list(blob) in [list(b1), list(b2)]:
                    color = "red"
                else:
                    color = "blue"
                y, x, r = blob
                c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
                ax.add_patch(c)
            if not pick:
                ax.set_axis_off()
            plt.tight_layout()
            plt.show()

        th1 = np.rad2deg(np.arctan2(b1[0] - dy2, b1[1] - dx2))
        th2 = np.rad2deg(np.arctan2(b2[0] - dy2, b2[1] - dx2))

        vprint("th1 th2: ", th1, th2)

        if np.abs(round(th1 - th2)) != 180:
            vprint("Incorrect blobs found, not setting tilt_dir")
            vprint(f"Angle found: {th1-th2:.2f}")
            self.tilt_dir = None
            return "Too many blobs found"
        else:
            vprint("Found blobs that seem correct.")
            ang = max(th1, th2)
            if ang >= 180:
                self.tilt_dir = ang - 180
            else:
                self.tilt_dir = ang
            if self.tilt_angle > 0:
                self.tilt_dir += 180
            if fix_tilt:
                vprint("Fixing tilt.")
                if bbox_im:
                    bbox_dig = None  # default bbox values
                else:
                    bbox_dig = bbox_digits  # bbox_dig < 0 means do not bbox
                self.fix_tilt(bbox_digits=bbox_dig)
        if show:
            if self.im is not None:
                im = self.im
            else:
                im = self.orig_im
            show_im(
                bbox(ndi.rotate(im, 90 + self.tilt_dir), digits=1),
                title="Rotated image. Skyrmion white should be on left.",
            )
        return "Two blobs found"

    def load_skyrms(self, centers, overwrite=False):
        if np.any(self.skyrm_list) and not overwrite:
            inp = "tmp"
            while inp != "y":
                inp = input("Skyrm list not empty, do you want to overwrite? (y/n)")
                inp = inp.strip().lower()
                if inp == "n":
                    return

        self.skyrm_list = []
        for i, center in enumerate(centers):
            y, x = center
            skyrm = Skyrm((y, x), imname=self.name, scale=self.scale)
            skyrm.index = i
            self.skyrm_list.append(skyrm)

        return

    def get_skyrm_averages(self):
        psi6abs_lst = []
        psi4abs_lst = []
        psi6cabs_lst = []
        psi4cabs_lst = []
        num_neighbors_lst = []
        vor_area_lst = []
        vor_circ_rad_lst = []
        num_not_edge = 0

        for skyrm in self.skyrm_list:
            if not skyrm.edge:
                psi6abs_lst.append(np.abs(skyrm.psi6))
                psi4abs_lst.append(np.abs(skyrm.psi4))
                psi6cabs_lst.append(np.abs(skyrm.psi6c))
                psi4cabs_lst.append(np.abs(skyrm.psi4c))
                num_neighbors_lst.append(len(skyrm.neighbors))
                vor_area_lst.append(skyrm.vor_area)
                vor_circ_rad_lst.append(skyrm.vor_circ[2])
                num_not_edge += 1

        self.skyrm_aves = {}
        self.skyrm_aves["psi6abs"] = [np.mean(psi6abs_lst), np.std(psi6abs_lst)]
        self.skyrm_aves["psi4abs"] = [np.mean(psi4abs_lst), np.std(psi4abs_lst)]
        self.skyrm_aves["psi6cabs"] = [np.mean(psi6cabs_lst), np.std(psi6cabs_lst)]
        self.skyrm_aves["psi4cabs"] = [np.mean(psi4cabs_lst), np.std(psi4cabs_lst)]
        self.skyrm_aves["num_neighbors"] = [
            np.mean(num_neighbors_lst),
            np.std(num_neighbors_lst),
        ]
        self.skyrm_aves["vor_area"] = [np.mean(vor_area_lst), np.std(vor_area_lst)]
        self.skyrm_aves["vor_circ_rad"] = [
            np.mean(vor_circ_rad_lst),
            np.std(vor_circ_rad_lst),
        ]
        self.skyrm_aves["im_scale"] = self.scale
        self.skyrm_aves["num_skyrms"] = [num_not_edge, len(self.skyrm_list)]
        self.skyrm_aves["im_dims"] = self.shape
        self.skyrm_aves["fft_skyrm_size"] = self.fft_skyrm_size

    def set_scale(self, scale, loading=False):
        """Change the scale of the images (nm/pix) in the relevant places.

        Args:
            scale (float): Scale of images in nm/pixel
            loading (bool): If loading the image then don't need to change the scale for
                each skyrm object (as they're already set) or adjust the fft skyrm size.

        Returns:
            None
        """
        old_scale = self.scale
        if self.axes_manager is not None:
            self.axes_manager.convert_units(units=["nm", "nm"])  # nm/pix
            self.axes_manager[0].scale = scale
            self.axes_manager[1].scale = scale

        if self.fft_skyrm_size is not None and not loading:
            new_fft_sk_size = (
                self.fft_skyrm_size[0] / old_scale * scale,
                self.fft_skyrm_size[1] / old_scale * scale,
            )
            self.fft_skyrm_size = new_fft_sk_size

        if self.skyrm_list is not None and not loading:
            for skyrm in self.skyrm_list:
                skyrm.im_scale = scale

        if self.skyrm_aves is not None and not loading:
            self.skyrm_aves["im_scale"] = scale
            new_fft_sk_size = (
                self.skyrm_aves["fft_skyrm_size"][0] / old_scale * scale,
                self.skyrm_aves["fft_skyrm_size"][1] / old_scale * scale,
            )
            self.skyrm_aves["fft_skyrm_size"] = new_fft_sk_size
            self.skyrm_aves["im_scale"] = scale

        self.scale = scale
        return

    # def rescale_Im(self, factor):
    #     """
    #     DEPRECATED - rescaling is now done NN side
    #     Rescale the image by factor, useful because the NN generally expects
    #     features to be similar in pixel size to the training data
    #     e.g. a 512x512 image with rescale_Im(2) will be 1024x1024
    #     TODO:
    #     the orig_im also gets rescaled, but it probably shouldn't. Better would
    #     be to have checks for self.rescale whenever doing anything with orig_im
    #     """
    #     self.rescale = factor
    #     if self.orig_im is not None:
    #         self.orig_im = skimage.transform.rescale(
    #             self.orig_im, factor, preserve_range=True
    #         )
    #     if self.full_im is not None:
    #         self.full_im = skimage.transform.rescale(
    #             self.full_im, factor, preserve_range=True
    #         )
    #     if self.im is not None:
    #         self.im = skimage.transform.rescale(self.im, factor, preserve_range=True)
    #         self.shape = self.im.shape
    #     self.set_scale(self.scale / factor)
    #     return

    def save_Im(self, directory=None, name=None, overwrite=False):
        """
        Save all attributes (but not images or things loaded from .dm3) as a .pkl
        """
        if directory is None:
            im_dir = os.path.split(self.filepath)[0]
            directory = os.path.join(im_dir, "Tilt_Ims")
            if not os.path.exists(directory):
                os.makedirs(directory)
        if name is None:
            name = self.name + ".pkl"

        self.modified = datetime.now()

        filename = os.path.abspath(os.path.join(directory, name))
        if os.path.isfile(filename):
            if overwrite:
                print(filename, " already exists. Overwriting.")
            else:
                print(f"File found at: {filename}")
                filename = overwrite_rename(filename)
                print(f"Saving as: {filename}")

        orig_im = np.copy(self.orig_im)
        full_im = np.copy(self.full_im) if self.full_im is not None else None
        im = np.copy(self.im) if self.im is not None else None
        axes_manager = deepcopy(self.axes_manager)
        metadata = deepcopy(self.metadata)
        skyrm_list = deepcopy(self.skyrm_list)

        # list of all Skyrm properties
        # probably would be fine to pickle the Skyrms... but better safe than sorry.
        self.skyrm_list = [vars(skyrm) for skyrm in self.skyrm_list]
        self.full_im = None
        self.orig_im = None
        self.im = None
        self.axes_manager = None
        self.metadata = None

        with open(filename, "wb") as output:  # overwrites
            dill.dump(self.__dict__, output, -1)

        self.full_im = full_im
        self.orig_im = orig_im
        self.im = im
        self.axes_manager = axes_manager
        self.metadata = metadata
        self.skyrm_list = skyrm_list

        print(f"Saved to {filename}")
        return filename

    @classmethod
    def load_pkl(cls, filepath, fast=False, **kwargs):
        """
        load Tilt_Im from .pkl storing the .__dict__ values
        """

        f, ext = os.path.splitext(filepath)

        with open(filepath, "rb") as load:
            im_dict = dill.load(load)
            Im = cls()
            Im.__dict__ = im_dict

        # populate the skyrm_list with Skyrms
        sk_dicts = np.copy(Im.skyrm_list)
        Im.skyrm_list = [Skyrm(_, _) for _ in sk_dicts]
        for skyrm, value in zip(Im.skyrm_list, sk_dicts):
            skyrm.__dict__ = value

        if not fast:
            load_scale = Im.scale  # scale gets reset with load_dm3()
            Im.load_dm3()
            Im.set_scale(load_scale, loading=True)
            if "analys" in str(filepath).lower() or "align" in str(filepath).lower():
                if "bbox" not in kwargs:
                    print(
                        "if this is an aligned dataset you might want to set 'bbox'=-1"
                    )
            bbox_dig = kwargs.get("bbox", None)  # None is default, picks value
            Im.fix_tilt(bbox_digits=bbox_dig)
            Im.crop_im(Im.crop)
            Im.filter_background()

        return Im

    def __repr__(self):
        return f"Tilt_Im(Name={self.name!r}, tilt_angle={self.tilt_angle})"


### deprecated
# def save_Tilt_Im_full(Im, directory=None, name=None):
#     # dill is so much nicer than pickle, doesnt get fucked by classes methods
#     # changing, but it is slower. definitely worth it.
#     #
#     # that said, the proper way to do this would be to save all parameters
#     # as json or something... and then have a loader function that initializes
#     # the new object and populates it. Really should never be saving objects,
#     # just the data (raw and extracted)
#     if directory is None:
#         im_dir = os.path.split(Im.filepath)[0]
#         directory = os.path.join(im_dir, "Tilt_Ims")
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#     if name is None:
#         name = Im.image_name + ".pkl"

#     ## Toss the image data/set to none
#     ## then pickle the whole thing
#     ## so only needs to reload the image data, and filter it, crop it
#     ## all of which should be repeatable, and the pickled object is small.
#     ## have to copy all the stuff so the final Im is the same, but whatever
#     if Im.full_im is not None:
#         full_im = np.copy(Im.full_im)
#     else:
#         full_im = None
#     orig_im = np.copy(Im.orig_im)
#     if Im.im is not None:
#         im = np.copy(Im.im)
#     else:
#         im = None
#     axes_manager = deepcopy(Im.axes_manager)
#     metadata = deepcopy(Im.metadata)

#     Im.full_im = None
#     Im.orig_im = None
#     Im.im = None
#     Im.axes_manager = None
#     Im.metadata = None

#     filename = os.path.join(directory, name)
#     if os.path.isfile(filename):
#         print(filename, " already exists. Overwriting.")
#     with open(filename, "wb") as output:  # overwrites
#         dill.dump(Im, output, -1)
#         # if this gives problems with people loading, switch to saving in older protocol
#         # (or even version 0 which is human readable)

#     Im.full_im = full_im
#     Im.orig_im = orig_im
#     Im.im = im
#     Im.axes_manager = axes_manager
#     Im.metadata = metadata

#     print(f"Saved to {filename}")
#     return filename

### Deprecated
# def load_Tilt_Im_full(filepath, fast=False, save_new=True, **kwargs):
#     """
#     loading a pickle (dill) of the full Tilt_Im object, as opposed to just
#     pickling the values, which is what the new save does
#     fast = True will not load any image data
#     """
#     f, ext = os.path.splitext(filepath)
#     fnew = f + "_new" + ext
#     loop = True
#     if os.path.isfile(fnew):
#         while loop:
#             usenew = input(
#                 "newer version of .pkl exists.\nDo you want to use it? (y/n): "
#             )
#             if usenew.lower() == "y":
#                 filepath = fnew
#                 loop = False
#             elif usenew.lower() == "n":
#                 loop = False

#     with open(filepath, "rb") as load:
#         Im = dill.load(load)
#         # taking care of legacy stuff so compatible with old version
#         resave = False
#         if not (
#             isinstance(Im.fft_skyrm_size, list)
#             or isinstance(Im.fft_skyrm_size, np.ndarray)
#             or isinstance(Im.fft_skyrm_size, tuple)
#         ):
#             Im.fft_skyrm_size = [Im.fft_skyrm_size, -1]
#             resave = True

#     if not fast:
#         Im.load_dm3()
#         bbox_dig = kwargs.get("bbox", None)  # None is default, picks value

#         if hasattr(Im, "tilt_axis"):  # fixing legacy names issues with tilt series
#             print("Old Tilt_Im object found, attmepting to get proper tilt_dir")
#             if Im.tilt_axis is not None:
#                 Im.tilt_axis = None
#                 Im.get_tilt_dir(
#                     show=False, min_sigma=35, max_sigma=50, threshold=2e6, pick=False
#                 )
#                 resave = True

#         Im.fix_tilt(bbox_digits=bbox_dig)
#         Im.crop_im(Im.crop)
#         Im.filter_background()

#     if hasattr(Im, "rescale"):
#         if Im.rescale is not None:
#             print(Im.rescale)
#             Im.rescale_Im(Im.rescale)

#     if resave and save_new:
#         im_dir = os.path.split(Im.filepath)[0]
#         directory = os.path.join(im_dir, "Tilt_Ims")
#         name = overwrite_rename(
#             os.path.join(directory, Im.image_name + "_new" + ".pkl")
#         )
#         print("name: ", name)
#         print(os.path.split(name))
#         save_Tilt_Im(Im, os.path.split(name)[0], name=os.path.split(name)[1])
#     return Im


# def load_update_Im(path, overwrite=False):
#     Im_old = load_Tilt_Im(path, fast=True)

#     old_keys = Im_old.__dict__.keys()

#     Im_new = Tilt_Im(Im_old.filepath, Im_old.image_name, Im_old.tilt_angle)
#     Im_new.get_tilt_dir(show=False, min_sigma = 35, max_sigma=50, threshold=2e6, pick=False)


#     for key in old_keys:
#         if key == 'tilt_axis':
#             continue
#         Im_new.__dict__[key] = Im_old.__dict__[key]

#     if overwrite:
#         spath = path
#     else:
#         spath = overwrite_rename(path)

#     save_Tilt_Im(Im_new, os.path.split(spath)[0], os.path.split(spath)[1])

#     return load_Tilt_Im(spath)
