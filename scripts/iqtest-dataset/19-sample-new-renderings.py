import numpy as np

from inversegraphics_generator.iqtest_objs import get_data_dir

ds = np.load(get_data_dir()+"/10000_baseline_data.npy")

print (ds.shape, ds.dtype, ds.max(), ds.min())
quit()

import matplotlib.pyplot as plt
for i in range(len(ds)):
    f, axarr = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(20, 5))

    for img in range(4):
        img_data = ds[i,img]/255
        img_data = np.swapaxes(np.swapaxes(img_data,0,2),0,1)
        print(img_data.min(), img_data.max(), img_data.mean())

        axarr[img].imshow(img_data)

    plt.tight_layout()
    plt.show()