from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
from .feature import Feature

class Hog(Feature):
    
    def __init__(self, num_processes, temp_dir, hdf5_path, bin_size):
        super().__init__("hog", num_processes, temp_dir, hdf5_path)
        self.bin_size = bin_size
        
    def describe(self, image):
        resized_image = resize(image,  (256, 256))
        feature, hog_image = hog(resized_image, 
                                 orientations=self.bin_size,
                                 pixels_per_cell=(16, 16),
                                 cells_per_block=(2, 2),
                                 visualize=True,
                                 feature_vector=True,
                                 multichannel=True)
        hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        return feature, hog_image
        
        
        
    def get_feature(self, image):
        feature, _ = self.describe(image)
        return feature
