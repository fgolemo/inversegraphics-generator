import os
import matplotlib.pyplot as plt
import numpy as np
from inversegraphics_generator.img_dataset import IqImgDataset
from inversegraphics_generator.iqtest_objs import get_data_dir

ds_train = IqImgDataset("/data/lisa/data/iqtest/iqtest-dataset-ambient.h5", "train/labeled")
ds_test = IqImgDataset("/data/lisa/data/iqtest/iqtest-dataset-ambient.h5", "test")

OUT = os.path.join(get_data_dir(),"sample-{traintest}-{idx}-{refans}.png")

def reshape(img):
    img = np.swapaxes(img, 0,2)
    img = np.swapaxes(img, 0,1)
    img *= 255
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

def sample(ds, idx, traintest):
    plt.imsave(OUT.format(traintest=traintest, refans="ref", idx=idx),reshape(ds[idx][0][0]))

    ans_idx = ds[idx][1].argmax()
    for ans in range(3):
        label = "ans"+str(ans+1)
        if ans == ans_idx:
            label += "c"
        plt.imsave(OUT.format(traintest=traintest, refans=label, idx=idx),reshape(ds[idx][0][1+ans]))


for i in range(10):
    sample(ds_train, i, "train")
    sample(ds_test, i, "test")





