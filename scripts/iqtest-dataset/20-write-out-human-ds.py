import os

import matplotlib.pyplot as plt
from scipy.misc import imresize

from inversegraphics_generator.human_dataset import HumanDataset
from inversegraphics_generator.iqtest_objs import get_data_dir

ds = HumanDataset()

FACTOR = 400
OUT_PATH = os.path.join(get_data_dir(), "..", "3diqtt", "imgs")

corrects = []

for i in range(len(ds)):
    (question, answers, correct) = ds[i]
    q_scaled = imresize(question, FACTOR)
    plt.imsave(os.path.join(OUT_PATH,"{}-q.png".format(i)), q_scaled)

    for a in range(3):
        a_scaled = imresize(answers[a], FACTOR)
        plt.imsave(os.path.join(OUT_PATH,"{}-a{}.png".format(i,a)), a_scaled)

    corrects.append(str(correct))

with open(os.path.join(OUT_PATH,"_correct.csv"),"w") as f:
    f.writelines(corrects)


(question, answers, correct) = ds.get(1)
q_scaled = imresize(question, FACTOR)
plt.imsave(os.path.join(OUT_PATH,"tutorial-q.png".format(i)), q_scaled)

for a in range(3):
    a_scaled = imresize(answers[a], FACTOR)
    plt.imsave(os.path.join(OUT_PATH,"tutorial-a{}.png".format(a)), a_scaled)







