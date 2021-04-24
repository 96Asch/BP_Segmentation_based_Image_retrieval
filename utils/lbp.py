from skimage import feature
import numpy as np
import h5py
import cv2
from .progressbar import update_progress


class LocalBinaryPatterns:

    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.numPoints + 3), range=(0, self.numPoints + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        return hist

    def dump_features(self, hdf5_path, resized, resize=299):
        db = h5py.File(hdf5_path, mode='r+')
        image_paths = db['id'][:]
        hists = []

        print("Extracting histograms from {} images".format(len(image_paths)))
        count = 0
        for image_path in image_paths:
            count += 1
            image = cv2.imread(image_path)
            if resized:
                image = cv2.resize(image, (resize, resize))
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hists.append(self.describe(gray_image))
            update_progress("Processed: {} | Progress".format(image_path), count, len(image_paths))

        hists = np.array(hists)
        db_features = db.create_dataset("lbp", shape=hists.shape, dtype="float")
        db_features[:] = hists
        db.close
