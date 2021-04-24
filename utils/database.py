import h5py
import glob


class Database:

    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path

    def create(self, hdf5_path, image_dir, image_formats={"jpg"}):
        self.hdf5_path = hdf5_path
        db = h5py.File(hdf5_path, mode='w')
        image_paths = []

        for image_format in image_formats:
            image_paths += glob.glob("{}/*.{}".format(image_dir, image_format))
        print("Creating dataset from {} images".format(len(image_paths)))

        db_image_ids = db.create_dataset("id", shape=(len(image_paths),), dtype=h5py.special_dtype(vlen=str))
        db_image_ids[:] = image_paths
        print("Finished creating dataset")
        db.close()

    def read(self, column):
        db = h5py.File(self.hdf5_path, mode="r")
        dataset = db[column][:].copy()
        db.close()
        return dataset
