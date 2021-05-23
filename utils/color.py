import numpy as np
import h5py
import cv2
from scipy.stats import skew
from .progressbar import update_progress
from .feature import Feature


class Color(Feature):
    
    def __init__(self, num_processes, temp_dir, hdf5_path, num_bins):
        super().__init__("rgb", num_processes, temp_dir, hdf5_path)
        self.num_bins = num_bins

    def describe(self, image, mask=[]):   
        bgr_planes = cv2.split(image)
        hist_range = (0, 255)
        weights = np.ones(bgr_planes[0].shape)
        if len(mask) > 0:
            weights = mask.astype(np.int)
        b_hist, b_bins = np.histogram(bgr_planes[0], self.num_bins, hist_range, density=True, weights=weights)
        g_hist, g_bins = np.histogram(bgr_planes[1], self.num_bins, hist_range, density=True, weights=weights)
        r_hist, r_bins = np.histogram(bgr_planes[2], self.num_bins, hist_range, density=True, weights=weights)
        
        return np.concatenate((b_hist, g_hist, r_hist)), b_bins
        
        

    def get_feature(self, image):
        feature, _ = self.describe(image)
        return feature
