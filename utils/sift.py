import matplotlib.pyplot as plt
from multiprocessing import Pool
from multiprocessing import cpu_count
from imutils import paths
import cv2
import numpy as np
import pickle
import os
import h5py
from .feature import Feature

class Sift(Feature):
    
    
    
    def __init__(self, num_processes, temp_dir, hdf5_path, num_octaves):
        super().__init__("sift", num_processes, temp_dir, hdf5_path)
        self.num_octaves = num_octaves
        
    def process(self, payload):
        print("[INFO] starting process {}".format(payload["id"]))
        features = {}
        sift = cv2.SIFT_create(nOctaveLayers=self.num_octaves)
        for index, image_path in enumerate(payload["input_paths"]):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            keypoints, descriptor = sift.detectAndCompute(image, None)
            features[image_path] = descriptor
                
            if (index % 50) == 0:
                print("Process {}: completed {} of {}".format(payload["id"], index, len(payload["input_paths"])))

        print("[INFO] process {} serializing features".format(payload["id"]))
        print(len(features))
        f = open(payload["output_path"], "wb")
        f.write(pickle.dumps(features))
        f.close()
    
    def dump(self, image_paths, overwrite=False):
        with h5py.File(self.hdf5_path, mode='r+') as db:
            if (self.name in db.keys()) and (overwrite == False):
                print("Dataset {} already exists in {}".format(self.name, self.hdf5_path))
                return
            
            payloads = []
            num_images_per_proc = len(image_paths) / float(self.num_processes)
            num_images_per_proc = int(np.ceil(num_images_per_proc))
            chunked_paths = list(self.chunk(image_paths, num_images_per_proc))
            
            self.remove_temp_files()
            
            # loop over the set chunked image paths
            for (i, i_paths) in enumerate(chunked_paths):
                output_path = os.path.sep.join([self.temp_dir, "proc_{}_{}.pickle".format(i, self.name)])
                data = {
                        "id": i,
                        "input_paths": i_paths,
                        "output_path": output_path
                        }
                payloads.append(data)
                                
            print("[INFO] launching pool using {} processes...".format(self.num_processes))
            pool = Pool(processes=self.num_processes)
            pool.map(self.process, payloads)

            print("[INFO] waiting for processes to finish...")
            pool.close()
            pool.join()
            print("[INFO] multiprocessing complete")
                                
            print("[INFO] combining features...")
            features = []

            for p in sorted(paths.list_files(self.temp_dir, validExts=(".pickle"))):
                data = pickle.loads(open(p, "rb").read())
               
                for (temp_path, temp_feature) in data.items():
                    data_name = "{}_{}".format(temp_path, self.name)
                    dataset = db.require_dataset(data_name, shape=temp_feature.shape, dtype="float")
                    
                    dataset[:] = temp_feature.astype(np.float32)                        
                    
                print("Appended {}".format(p))
            db.require_dataset(self.name, shape=(1, 1), dtype="float")
        print('Features Dumped succesfully')
