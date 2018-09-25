import os

import h5py
import numpy as np

from inversegraphics_generator.iqtest_objs import get_data_dir

with h5py.File(os.path.join(get_data_dir(), "test.h5"), "w") as f:
    train = f.create_group("train")
    test = f.create_group("test")
    val = f.create_group("val")

    labeled = train.create_group("labeled")
    unlabeled = train.create_group("unlabeled")

    labeled_q = labeled.create_dataset("input",data=np.zeros((10,4,128,128,3),dtype=np.float32))
    labeled_a = labeled.create_dataset("output",data=np.zeros(10),dtype=np.uint8)