import os
import numpy as np
import h5py
from torch.utils.data import Dataset

from inversegraphics_generator.iqtest_objs import get_data_dir


class IqImgDataset(Dataset):
    def __init__(self, h5_path, group):
        assert group in ["train/labeled", "train/unlabeled", "test", "val"]
        self.file_handle = h5py.File(h5_path, 'r')
        self.group = group
        self.data = self.file_handle[group]

    def __len__(self):
        return len(self.data["input"])

    def __getitem__(self, idx):
        # out = {"input": self.data["input"][idx], "output": None}
        # if self.group in ["train/labeled", "test", "val"]:
        #     out["output"] = self.data["output"][idx]
        # return out

        answer = None
        if self.group in ["train/labeled", "test", "val"]:
            answer = np.zeros(3,dtype=np.float32)
            answer[self.data["output"][idx]] = 1 # one-hot encoding
            # print (self.data["output"][idx], answer)

        question = np.swapaxes(
            np.swapaxes(
                self.data["input"][idx],
                1, 3),
            2, 3
        )

        return (question, answer)


if __name__ == '__main__':
    ds = IqImgDataset(os.path.join(get_data_dir(), "test.h5"), "train/labeled")

    print(len(ds))

    question, answer = ds[0]

    print(question.shape)
    print(answer)
