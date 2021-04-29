import h5py
import glob
import os.path
from os import path


class Database:

    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path

    def create(self, hdf5_path, image_paths, image_formats={"jpg"}):
        if path.exists(hdf5_path):
            print("Database exists at {}".format(hdf5_path))
            return
            
        self.hdf5_path = hdf5_path
        with h5py.File(hdf5_path, mode='w') as db:
            db_image_ids = db.require_dataset("id", shape=(len(image_paths),), dtype=h5py.special_dtype(vlen=str))
            db_image_ids[:] = image_paths
            print("Finished creating dataset")

    def read(self, column):
        db = h5py.File(self.hdf5_path, mode="r")
        dataset = db[column][:].copy()
        db.close()
        return dataset
