#!/usr/bin/env python3

import os
import colorsys

import cv2
import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class ColorPredicate:

    def __init__(self, name, images_path, n_max=10):
        self.name = name
        self._total_pixel_count = 0
        self.images = self.load_images(images_path, n_max)
        self.masks = [255 * np.ones(image.shape[:2], np.uint8)
                      for image in self.images]
        self._histogram_channels = None
        self._histogram_color_space = None
        self._bins = None
        self._grid = None
        self._ch_indexes = None
        self._target_histogram = None
        self._background_histogram = None
        self._gaussian_smoothed_histogram = None
        self._color_predicate = None

        self.color_spaces = {
            'hsv': cv2.COLOR_BGR2HSV
        }
        self.ch_ranges = {
            'b': (0, 256), 'g': (0, 256), 'r': (0, 256),
            'h': (0, 180), 's': (0, 256), 'v': (0, 256)
        }

    def load_images(self, path, n_max):
        """Load and return a list of up to `n_max` images from `path`."""
        images_list = []
        n_max = min(n_max, len(os.listdir(path)))
        for filename in sorted(os.listdir(path))[:n_max]:
            image = cv2.imread(os.path.join(path, filename))
            if image is not None:
                images_list.append(image)
                self._total_pixel_count += image.shape[0] * image.shape[1]

        return images_list

    def load_masks(self, path):
        """Load and return a list of image masks from path."""
        masks_list = []
        n_images = len(self.images)
        n_masks = len(os.listdir(path))
        if n_masks >= len(self.images):
            for filename in sorted(os.listdir(path))[:n_images]:
                mask_gray = cv2.imread(os.path.join(path, filename), 0)
                ret, mask = cv2.threshold(mask_gray, 127, 255,
                                          cv2.THRESH_BINARY)
                if mask is not None:
                    masks_list.append(mask)
            self.masks = masks_list
        else:
            print(f'Directory must contain at least {n_images} image files, '
                  f'but only {n_masks} were provided. Masks will be ignored.')

    @staticmethod
    def sample_pixels(target_pixels, bg_pixels, target_sr, bg_rate):
        """Take a random sample of target and background pixels.

        Parameters
        ----------
        target_pixels : numpy.ndarray
            Array of pixels from target region.
        bg_pixels : numpy.ndarray
            Array of pixels from background region.
        target_sr : int or float
            Target pixels sample rate (percentage of total target pixels).
        bg_rate : int or float
            Ratio of background to target pixels.
            A value of 1.0 means equivalent distribution.

        Returns
        -------
            target_pixels_sample : numpy.ndarray
                Array of random samples from target region.
            bg_pixels_sample : numpy.ndarray
                Array of random samples from background region.

        """
        n_target_pixels, n_bg_pixels = len(target_pixels), len(bg_pixels)
        target_samples = n_target_pixels * target_sr

        if n_bg_pixels > 0:

            n_bg_samples = target_samples * bg_rate
            target_bg_ratio = n_target_pixels / n_bg_pixels

            if n_bg_samples > n_bg_pixels:
                target_sr = n_bg_pixels / (n_target_pixels * bg_rate)

            bg_sr = target_bg_ratio * target_sr * bg_rate

            indexes_bg_samples = np.random.choice([0, 1],
                                                  size=n_bg_pixels,
                                                  p=[(1 - bg_sr), bg_sr])

            bg_pixels_sample = bg_pixels[indexes_bg_samples == 1]
        else:
            bg_pixels_sample = bg_pixels

        indexes_target_samples = np.random.choice([0, 1],
                                                  size=n_target_pixels,
                                                  p=[1 - target_sr, target_sr])

        target_pixels_sample = target_pixels[indexes_target_samples == 1]

        return target_pixels_sample, bg_pixels_sample

    def create_multidimensional_histogram(self, color_space='bgr',
                                          ch_indexes=(0, 1, 2),
                                          bins=(8, 8, 8),
                                          target_sr=1.0,
                                          bg_rate=1.0):
        """Create a multidimensional histogram of instance's images.

        Color space can be either RGB or HSV. Dimension is set according
        to `ch_indexes` length. Sampling can be specified.

        Parameters
        ----------
        color_space : str, optional
            Histogram color space. Accepts `bgr` (default) or `hsv`.
        ch_indexes : tuple, optional
            Sequence of histogram channel indexes. Values refer to
            `color_space` string order. E.g, use (0, 2) to create a
            2D histogram of channels b and r.
        bins : tuple, optional
            Sequence of histogram bins. Must be of same length of `ch_indexes`.
        target_sr : int or float
            Target pixels sample rate (percentage of total target pixels).
        bg_rate : int or float
            Ratio of background to target pixels. A value of 1.0 means
            equivalent distribution.

        Returns
        -------
        self._target_histogram : numpy.ndarray
            2D or 3D histogram of sampled target pixels
        self._bg_histogram : numpy.ndarray
            2D or 3D histogram of samples background pixels

        """
        print('Computing histogram...', end=' ')

        target_pixels_per_image, bg_pixels_per_image = [], []

        if sorted(ch_indexes) in ([0, 1], [0, 2], [1, 2], [0, 1, 2]):
            self._histogram_channels = [color_space[i] for i in ch_indexes]
            hist_range = [self.ch_ranges[ch] for ch in self._histogram_channels]
        else:
            raise ValueError('Parameter "ch_indexes" must be a sequence '
                             'of unique integers between 0 and 2')

        for image, mask in zip(self.images, self.masks):
            if color_space != 'bgr':
                image = cv2.cvtColor(image, self.color_spaces[color_space])
            target_pixels_per_image.append(image[mask > 0])
            bg_pixels_per_image.append(image[~mask > 0])

        target_pixels = np.concatenate(target_pixels_per_image)
        bg_pixels = np.concatenate(bg_pixels_per_image)

        target_samples, bg_samples = self.sample_pixels(target_pixels,
                                                        bg_pixels,
                                                        target_sr,
                                                        bg_rate)

        self._target_histogram, _ = np.histogramdd(target_samples[:, ch_indexes],
                                                   bins=bins,
                                                   range=hist_range)

        self._background_histogram, _ = np.histogramdd(bg_samples[:, ch_indexes],
                                                       bins=bins,
                                                       range=hist_range)
        self._bins = bins
        self._histogram_color_space = color_space
        self._ch_indexes = ch_indexes

        print('Done!')

        return self._target_histogram, self._background_histogram

    def pdf(self, mean, cov, domain):
        """Multidimensional probability density function."""
        pdf = multivariate_normal.pdf(domain, mean=mean, cov=cov)
        pdf = pdf.reshape(self._bins)

        return pdf

    def create_gaussian_smoothed_histogram(self,
                                           t_amp=1.0,
                                           t_cov=0.05,
                                           bg_amp=1.0,
                                           bg_cov=0.025,
                                           threshold=0.01,
                                           norm=True):
        """Create a 2D or 3D gaussian-smoothed histogram.

        A gaussian-smoothed histogram is built from target and background
        pixels according to [1]: for each pixel in target region, a normal
        distribution centered at its position is added to the histogram;
        similarly, for each pixel at background, a normal distribution is
        subtracted. Finally, thresholding is applied: color frequencies below
        threshold times maximum frequency are set to zero.

        [1] `Finding skin in color images`, R. Kjeldsen and J. Kender.
        Proceedings of the Second International Conference on Automatic Face
        and Gesture Recognition, 1996. DOI:10.1109/AFGR.1996.557283

        Parameters
        ----------
        t_amp : float, optional
            Amplitude of target's normal distribution. Default is 1.0.
        t_cov : float, optional
            Covariance of target's normal distribution. Default is 0.05.
        bg_amp : float, optional
            Amplitude of background's normal distribution. Default is 1.0.
        bg_cov : float, optional
            Covariance of background's normal distribution. Default is 0.025.
        threshold : float, optional
            Color frequencies below threshold times maximum frequency are
            set to zero. Default is 0.01.
        norm : bool, optional
            When True, histogram is normalized by maximum frequency. Default
            is True.

        Returns
        -------
        self._gaussian_smoothed_histogram : numpy.ndarray
            2D or 3D gaussian-smoothed histogram.

        """
        print('Generating gaussian-smoothed histogram...', end=' ')

        self._grid = np.mgrid[tuple([slice(0, b) for b in self._bins])]
        domain = np.column_stack([axis.flat for axis in self._grid])
        gauss_sum = np.zeros(self._bins, dtype=np.float32)

        t_cov = t_cov * min(self._bins)
        bg_cov = bg_cov * min(self._bins)

        t_hist = self._target_histogram
        bg_hist = self._background_histogram

        for pos in np.argwhere(t_hist):
            pdf = self.pdf(pos, t_cov, domain) * t_amp
            gauss_sum += pdf * t_hist[tuple(pos)]
        for pos in np.argwhere(bg_hist):
            pdf = - self.pdf(pos, bg_cov, domain) * bg_amp
            gauss_sum += pdf * bg_hist[tuple(pos)]

        gauss_sum[gauss_sum < threshold * np.max(gauss_sum)] = 0

        if norm:
            gauss_sum = gauss_sum / np.max(gauss_sum)

        self._gaussian_smoothed_histogram = gauss_sum

        print('Done!')

        return self._gaussian_smoothed_histogram

    def create_color_predicate(self, threshold=0, save=False, filename='color_predicate'):
        """Create a color predicate from gaussian-smoothed histogram.

        Parameters
        ----------
        threshold : int or float, optional
            Histogram frequencies above threshold are set to one; frequencies
            below threshold are set to zero. Default is 0.
        save : bool, optional
            If true, color predicate is saved as a numpy array. Default is
            False.
        filename : str, optional
            Color predicate file name. Default is `color_predicate`

        Returns
        -------
        color_predicate : numpy.ndarray
            Color predicate with the same dimension as the histogram.

        """
        color_predicate = self._gaussian_smoothed_histogram.copy()
        color_predicate[color_predicate > threshold] = 1
        color_predicate[color_predicate <= threshold] = 0

        if save:
            np.save(filename, color_predicate)

        return color_predicate

    def plot_gaussian_smoothed_histogram(self, figsize=(8, 8), dpi=75, save=False):
        """Plot a 2D or 3D gaussian-smoothed histogram.

        When 2D, creates a pseudocolor histogram; when 3D, each bin is
        represented by a circle with size proportional to its frequency.

        Parameters
        ----------
        figsize : tuple, optional
            Matplotlib's `figsize` parameter. Default is (8, 8).
        dpi : int, optional
            Matplotlib's `dpi` parameter. Default is 75.
        save : bool, optional
            When true, saves the plot as a png file.

        """
        print('Plotting gaussian smoothed histogram...', end=' ')

        grid = self._grid
        ranges = self.ch_ranges
        bins = self._bins
        channels = self._histogram_channels
        histogram = self._gaussian_smoothed_histogram
        color_space = self._histogram_color_space
        axis = [(ranges[ch][1] / bins[i]) * grid[i] + (ranges[ch][1] / bins[i]) / 2
                for i, ch in enumerate(channels)]

        if histogram.ndim == 3:

            colors = np.vstack((axis[0].flatten() / ranges[channels[0]][1],
                                axis[1].flatten() / ranges[channels[1]][1],
                                axis[2].flatten() / ranges[channels[2]][1])).T

            colors = colors[:, tuple([channels.index(ch) for ch in color_space])]

            if color_space == 'hsv':
                colors = np.array([colorsys.hsv_to_rgb(color[0], color[1], color[2])
                                   for color in colors])
            elif color_space == 'bgr':
                colors = colors[:, ::-1]

            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_subplot(111, projection='3d')
            ax.title.set_position([0.5, 1.1])
            ax.set_title(f'3D Color Histogram - '
                         f'{channels[0].title()} x '
                         f'{channels[1].title()} x '
                         f'{channels[2].title()}', fontsize=16)

            ax.xaxis.set_tick_params(labelsize=8)
            ax.yaxis.set_tick_params(labelsize=8)
            ax.zaxis.set_tick_params(labelsize=8)

            ax.set_xlim(ranges[channels[0]][0], ranges[channels[0]][1])
            ax.set_ylim(ranges[channels[1]][0], ranges[channels[1]][1])
            ax.set_zlim(ranges[channels[2]][0], ranges[channels[2]][1])

            ax.set_xlabel(channels[0].title(), fontsize=12)
            ax.set_ylabel(channels[1].title(), fontsize=12)
            ax.set_zlabel(channels[2].title(), fontsize=12)

            ax.view_init(azim=45)

            ax.scatter(axis[0], axis[1], axis[2],
                       s=histogram * 1000,
                       c=colors)

            if save:
                ch_str = channels[0] + channels[1] + channels[2]
                plt.savefig(f'{self.name}_3d_{ch_str}_histogram.png')

            plt.show()

        if self._gaussian_smoothed_histogram.ndim == 2:

            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_subplot(111)
            ax.set_aspect('equal')
            ax.set_title(f'2D Color Histogram - '
                         f'{channels[0].title()} x '
                         f'{channels[1].title()}')

            ax.set_xlabel(channels[0].title(), fontsize=12)
            ax.set_ylabel(channels[1].title(), fontsize=12, rotation=0)

            h = ax.pcolormesh(axis[0], axis[1], histogram)
            fig.colorbar(h, ax=ax)

            if save:
                ch_str = channels[0] + channels[1]
                plt.savefig(f'{self.name}_2d_{ch_str}_histogram.png')

            plt.show()

        print('Done!')

    @property
    def total_pixel_count(self):
        return self._total_pixel_count

    @property
    def gaussian_smoothed_histogram(self):
        return self._gaussian_smoothed_histogram

    @property
    def true_pixels_histogram(self):
        return self._target_histogram

    @property
    def false_pixels_histogram(self):
        return self._background_histogram

    @property
    def color_predicate(self):
        return self._color_predicate

    def __str__(self):
        description = f'''
        {self.name.title()} color predicate.

        Images: {len(self.images)}
        Bins: {self._bins}
        Color Space: {self._histogram_color_space}
        Channels: {self._ch_indexes}
        '''

        return description
