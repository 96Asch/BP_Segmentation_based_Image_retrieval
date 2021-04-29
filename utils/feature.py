from multiprocessing import Pool
from multiprocessing import cpu_count
from imutils import paths
import cv2
import numpy as np
import pickle
import os
import h5py
from .progressbar import update_progress
from abc import ABC, abstractmethod

class Feature:
    
    def __init__(self, name, num_processes, temp_dir, hdf5_path):
        self.num_processes = num_processes
        self.name = name
        self.temp_dir = temp_dir
        self.hdf5_path = hdf5_path
        self.make_temp_dir(temp_dir)
        
    
    def make_temp_dir(self, temp_dir):
        path = os.path.join(os.getcwd(), temp_dir)
        if os.path.isdir(path) == False:
            os.mkdir(path)
            print("Created Directory {}".format(temp_dir))
        
    def remove_temp_files(self):
        filelist = [ f for f in os.listdir(self.temp_dir) if f.endswith(".pickle") ]
        cwd_temp_dir = "{}/{}".format(os.getcwd(), self.temp_dir)
        for f in filelist:
            os.remove(os.path.join(cwd_temp_dir, f))
        print("Cleaned temp files in {}".format(cwd_temp_dir))
            
    def chunk(self, array, num_chunks):
        for i in range(0, len(array), num_chunks):
            yield array[i: i + num_chunks]
            
    @abstractmethod       
    def get_feature(self, image):
        pass
        
    def process(self, payload):
        print("[INFO] starting process {}".format(payload["id"]))
        features = {}
        for image_path in payload["input_paths"]:
            image = cv2.imread(image_path)
            features[image_path] = self.get_feature(image)

        print("[INFO] process {} serializing features".format(payload["id"]))
        f = open(payload["output_path"], "wb")
        f.write(pickle.dumps(features))
        f.close()
    
    def dump(self, image_paths):
        with h5py.File(self.hdf5_path, mode='r+') as db:
            if self.name in db.keys():
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
                    features.append(temp_feature)    
                print("Appended {}".format(p))
                                
            features = np.array(features)
            db_features = db.require_dataset(self.name, shape=features.shape, dtype="float")
            db_features[:] = features
        print('Features Dumped succesfully')