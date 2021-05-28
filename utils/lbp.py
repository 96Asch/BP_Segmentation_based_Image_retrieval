from skimage import feature
import numpy as np
import h5py
import cv2
from .feature import Feature


class LocalBinaryPatterns(Feature):
        
    def __init__(self, num_processes, temp_dir, hdf5_path, num_points, radius, eps=1e-7):
        super().__init__("lbp", num_processes, temp_dir, hdf5_path)
        self.num_points = num_points
        self.radius = radius
        self.eps = eps

    def describe(self, image, masked=[]):
        msk = np.ones((image.shape[0], image.shape[1])).astype(np.bool)
        if len(masked) > 0:
            msk = masked.astype(np.bool)
        
        lbp = feature.local_binary_pattern(image, self.num_points, self.radius, method="uniform")
        (hist, bins) = np.histogram(lbp[msk].ravel(), 
                                 bins=(self.num_points + 2), 
                                 range=(0, self.num_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + self.eps)
        return hist, bins
    
    def get_feature(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist, _ = self.describe(gray_image)
        return hist