import numpy as np
import h5py
import cv2
from scipy.stats import skew
from .progressbar import update_progress
from .feature import Feature


class Color(Feature):
    
    def __init__(self, num_processes, temp_dir, hdf5_path):
        super().__init__("color", num_processes, temp_dir, hdf5_path)

    def calc_mean(self, hsv_masked):
        return np.ma.mean(hsv_masked)

    def calc_variance(self, hsv_masked):
        return np.ma.var(hsv_masked)

    def calc_std(self, hsv_masked):
        return np.ma.std(hsv_masked)

    def calc_skewness(self, hsv_masked):
        return skew(hsv_masked)

    def normalize(self, hsv_masked):
        if np.ma.sum(hsv_masked) == 0:
            return hsv_masked
        min = np.ma.min(hsv_masked)
        max = np.ma.max(hsv_masked)
        return (hsv_masked - min) / (max - min)

    def calc_moment(self, hsv_matrix):
        mean = [self.calc_mean(hsv_matrix[i]) for i in range(3)]
        variance = [self.calc_variance(hsv_matrix[i]) for i in range(3)]
        std = [self.calc_std(hsv_matrix[i]) for i in range(3)]
        skewness = [self.calc_skewness(hsv_matrix[i]) for i in range(3)]
        color_vector = zip(mean, variance, std, skewness)
        color_vector = np.array([i for tuple in color_vector for i in tuple])
        return color_vector
    
    
    def describe(self, hsv_image, mask=None):
        if mask is None:
            mask = np.zeros((hsv_image.shape[0], hsv_image.shape[1]))
        mask = np.ravel(mask)
        h_channel = np.ma.array(np.ravel(hsv_image[:, :, 0]), mask=mask)
        s_channel = np.ma.array(np.ravel(hsv_image[:, :, 1]), mask=mask)
        v_channel = np.ma.array(np.ravel(hsv_image[:, :, 2]), mask=mask)
        return [self.normalize(h_channel), self.normalize(s_channel), self.normalize(v_channel)]
    
    def get_feature(self, image):
        return self.calc_moment(self.describe(image))
