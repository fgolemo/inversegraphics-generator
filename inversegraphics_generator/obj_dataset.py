import os
import time

import numpy as np
import tempfile

from inversegraphics_generator.obj_generator import ObjGenerator
from inversegraphics_generator.iqtest_objs import get_data_dir

MAX_GRID = 8


class IqObjDataset(object):
    def __init__(self, path):
        ds = np.load(path)
        self.train = ds["train"]
        self.test = ds["test"]
        self.val = ds["val"]
        print("dataset loaded:", path)
        self.og = ObjGenerator(MAX_GRID, 1.0)
        self.paths = []
        self.sample_idx = [0, 0, 0, 0]  # corresponding to train labeled, train unlabeled, test, val

    def _grid_to_file(self, grid, folder, idx, filepattern="{:06d}.obj"):
        v, f = self.og.grid_to_cubes(grid)
        out_path = os.path.join(folder.name, filepattern.format(idx))
        self.og.write_obj(grid, v, f, out_path)

    def sample_unordered(self, dataset, sidx, n=1):
        # make tmp dir
        folder = tempfile.TemporaryDirectory()
        self.paths.append(folder)

        # get N samples and write to disk
        for idx in range(n):
            self.sample_idx[sidx] += 1
            if self.sample_idx[sidx] == len(dataset):
                self.sample_idx[sidx] = 0

            self._grid_to_file(dataset[self.sample_idx[sidx], 0], folder, self.sample_idx[sidx])

        # return path
        return folder.name

    def sample_qa(self, dataset, sidx, n=1):
        # make tmp dir
        folder = tempfile.TemporaryDirectory()
        self.paths.append(folder)

        # get N samples and write to disk
        for idx in range(n):
            self.sample_idx[sidx] += 1
            if self.sample_idx[sidx] == len(dataset):
                self.sample_idx[sidx] = 0

            self._grid_to_file(dataset[self.sample_idx[sidx], 0], folder, self.sample_idx[sidx], "{:06d}-ref.obj")
            self._grid_to_file(dataset[self.sample_idx[sidx], 1], folder, self.sample_idx[sidx], "{:06d}-ans1.obj")
            self._grid_to_file(dataset[self.sample_idx[sidx], 2], folder, self.sample_idx[sidx], "{:06d}-ans2.obj")
            self._grid_to_file(dataset[self.sample_idx[sidx], 3], folder, self.sample_idx[sidx], "{:06d}-ans3.obj")

        # return path
        return folder.name

    def get_training_samples_unordered(self, n=1):
        return self.sample_unordered(self.train, 1, n)

    def get_training_questions_answers(self, n=1):
        return self.sample_qa(self.train, 0, n)

    def get_testing_questions_answers(self, n=1):
        return self.sample_qa(self.test, 2, n)

    def get_validation_questions_answers(self, n=1):
        return self.sample_qa(self.val, 3, n)

    def cleanup(self):
        for tmp in self.paths:
            tmp.cleanup()


if __name__ == '__main__':
    # iq = IqDataset(os.path.expanduser("~/data/ig/iqtest-v1.npz"))
    iq = IqObjDataset(os.path.join(get_data_dir(), "iqtest-v1.npz"))
    print(iq.train.shape)
    print(iq.test.shape)
    print(iq.val.shape)

    print(iq.get_training_samples_unordered(5))
    print(iq.get_training_questions_answers(5))
    print(iq.get_testing_questions_answers(5))
    print(iq.get_validation_questions_answers(5))

    time.sleep(3000)  # use this time to do `ls` on the directories printed above or pull them into Blender
    iq.cleanup()
