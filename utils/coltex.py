import math
import numpy as np
import cv2
import h5py
from .progressbar import update_progress
from .feature import Feature


class Coltex(Feature):
    
    def __init__(self, num_processes, temp_dir, hdf5_path, quantization_hue, quantization_int):
        super().__init__("coltex", num_processes, temp_dir, hdf5_path)
        self.quantization_hue = quantization_hue
        self.quantization_int = quantization_int

    def calculate_weight_hue(self, saturation, intensity):
        if intensity == 0:
            return 0
        else: 
            return saturation**(0.1 * ((255 / intensity) ** 0.85) )

    def calculate_weight_int(self, saturation, intensity):
        return 1 - self.calculate_weight_hue(saturation, intensity)

    def describe(self, hsv_image):
        
        n1 = math.ceil(((2 * math.pi) / self.quantization_hue) + 1)
        n2 = math.ceil((255 / self.quantization_int) + 1)
        N = n1 + n2

        DIAG = np.zeros((N, N))
        HORI = np.zeros((N, N))
        VERT = np.zeros((N, N))

        for i in range(hsv_image.shape[0]):
            for j in range(hsv_image.shape[1]):

                hue_main, sat_main, int_main = hsv_image[i][j][0], hsv_image[i][j][1], hsv_image[i][j][2]
                hue_diag, sat_diag, int_diag = 0, 0, 0
                hue_vert, sat_vert, int_vert = 0, 0, 0
                hue_hori, sat_hori, int_hori = 0, 0, 0
                
                if (i + 1) < hsv_image.shape[0] and (j + 1) < hsv_image.shape[1]:
                    hue_diag, sat_diag, int_diag = hsv_image[i+1][j+1][0], hsv_image[i+1][j+1][1], hsv_image[i+1][j+1][2]
                elif (i + 1) < hsv_image.shape[0]:
                    hue_vert, sat_vert, int_vert = hsv_image[i+1][j][0], hsv_image[i+1][j][1], hsv_image[i+1][j][2]
                elif (j + 1) < hsv_image.shape[1]:
                    hue_hori, sat_hori, int_hori = hsv_image[i][j+1][0], hsv_image[i][j+1][1], hsv_image[i][j+1][2]
                    

                wh_main, wi_main = self.calculate_weight_hue(sat_main, int_main), self.calculate_weight_int(sat_main, int_main)
                wh_diag, wi_diag = self.calculate_weight_hue(sat_diag, int_diag), self.calculate_weight_int(sat_diag, int_diag)
                wh_vert, wi_vert = self.calculate_weight_hue(sat_vert, int_vert), self.calculate_weight_int(sat_vert, int_vert)
                wh_hori, wi_hori = self.calculate_weight_hue(sat_hori, int_hori), self.calculate_weight_int(sat_hori, int_hori)

                phQh = math.ceil(hue_main / self.quantization_hue)
                qhQh = math.ceil(hue_diag / self.quantization_hue)
                nqiQi = math.ceil(n1 + (int_diag / self.quantization_int))
                npiQi = math.ceil(n1 + (int_main / self.quantization_int))

                DIAG[phQh][qhQh] += wh_main + wh_diag
                DIAG[phQh][nqiQi] +=  wh_main + wi_diag
                DIAG[npiQi][qhQh] += wi_main + wh_diag
                DIAG[npiQi][nqiQi] += wi_main + wh_diag

                qhQh = math.ceil(hue_vert / self.quantization_hue)
                nqiQi = math.ceil(n1 + (int_vert / self.quantization_int))

                VERT[phQh][qhQh] += wh_main + wh_vert
                VERT[phQh][nqiQi] +=  wh_main + wi_vert
                VERT[npiQi][qhQh] += wi_main + wh_vert
                VERT[npiQi][nqiQi] += wi_main + wh_vert

                qhQh = math.ceil(hue_hori / self.quantization_hue)
                nqiQi = math.ceil(n1 + (int_hori / self.quantization_int))

                HORI[phQh][qhQh] += wh_main + wh_hori
                HORI[phQh][nqiQi] +=  wh_main + wi_hori
                HORI[npiQi][qhQh] += wi_main + wh_hori
                HORI[npiQi][nqiQi] += wi_main + wh_hori
                
        return np.concatenate((DIAG.ravel(), VERT.ravel(), HORI.ravel()))
    
    def get_feature(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return self.describe(hsv_image)            