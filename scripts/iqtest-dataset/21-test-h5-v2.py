import numpy as np
import h5py
from tqdm import tqdm
from inversegraphics_generator.iqtest_objs import get_data_dir

OUT_PATH = get_data_dir() + "/3diqtt-v2-x.h5"

f = h5py.File(OUT_PATH, 'r')

print (list(f.keys()))


# dset = f['mydataset']


