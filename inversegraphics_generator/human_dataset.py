import os
import numpy as np
import h5py
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from inversegraphics_generator.iqtest_objs import get_data_dir

default_path = os.path.join(get_data_dir(),"..","3diqtt","human-test.h5")

class HumanDataset(Dataset):
    def __init__(self, h5_path=default_path):
        file_handle = h5py.File(h5_path, 'r')
        self.references = file_handle["references"]
        self.answers = file_handle["answers"]
        self.correct = file_handle["correct"]

    def __len__(self):
        return len(self.correct)

    def __getitem__(self, idx):
        return (self.references[idx,:,:,:3], self.answers[idx,:,:,:,:3], self.correct[idx])


if __name__ == '__main__':
    ds = HumanDataset()

    print(len(ds))

    question, answers, correct = ds[0]

    print(question.shape)
    print(answers.shape)
    print(correct)

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    ax1.imshow(question)
    ax2.imshow(answers[0])
    ax3.imshow(answers[1])
    ax4.imshow(answers[2])

    plt.tight_layout()
    plt.show()
