import os
import matplotlib.pyplot as plt
import h5py
from inversegraphics_generator.iqtest_objs import get_data_dir
import numpy as np

OUT_PATH = os.path.join(get_data_dir(), "..", "3diqtt", "human-test.h5")

questions = []
answers = []
correct = []

folder = os.path.join(get_data_dir(), "..", "3diqtt")

for i in range(100):
    ref_file = os.path.join(folder, "sample-test-{}-ref.png".format(i))
    if os.path.isfile(ref_file):
        questions.append(plt.imread(ref_file)[:, :, :3])

        answers_buf = []
        for a in range(3):
            ans_file = os.path.join(folder, "sample-test-{}-ans{}".format(i, a + 1))

            if not os.path.isfile(ans_file + ".png"):
                correct.append(a)
                ans_file += "c"
            answers_buf.append(plt.imread(ans_file + ".png")[:, :, :3])
        answers.append(answers_buf)

from random import shuffle

x = list(range(len(questions)))
shuffle(x)

with h5py.File(OUT_PATH, "w") as f:
    f.create_dataset("references", data=np.array(questions)[x])
    f.create_dataset("answers", data=np.array(answers)[x])
    f.create_dataset("correct", data=np.array(correct)[x])
