import os
import pickle
import numpy as np
from tqdm import tqdm

from inversegraphics_generator.obj_generator import ObjGenerator
from iqtest_objs import get_data_dir

OUT = os.path.join(get_data_dir(), "aggregated.pkl")

SNEK_LEN_1 = 11
GRID_SIZE_1 = 5
OUT_PATH_1 = os.path.join(get_data_dir(), "list-v2-snek_{}-grid_{}-final.pkl".format(SNEK_LEN_1, GRID_SIZE_1))

SNEK_LEN_2 = 13
GRID_SIZE_2 = 5
OUT_PATH_2 = os.path.join(get_data_dir(), "list-v2-snek_{}-grid_{}-final.pkl".format(SNEK_LEN_2, GRID_SIZE_2))

SNEK_LEN_3 = 14
GRID_SIZE_3 = 6
OUT_PATH_3 = os.path.join(get_data_dir(), "list-v2-snek_{}-grid_{}-final.pkl".format(SNEK_LEN_3, GRID_SIZE_3))

SNEK_LEN_4 = 15
GRID_SIZE_4 = 8
OUT_PATH_4 = os.path.join(get_data_dir(), "list-v2-snek_{}-grid_{}-final.pkl".format(SNEK_LEN_4, GRID_SIZE_4))

og1 = ObjGenerator(GRID_SIZE_1, 1.0)
og2 = ObjGenerator(GRID_SIZE_2, 1.0)
og3 = ObjGenerator(GRID_SIZE_3, 1.0)
og4 = ObjGenerator(GRID_SIZE_4, 1.0)

with open(OUT_PATH_1, "rb") as output_file:
    final1 = pickle.load(output_file)

with open(OUT_PATH_2, "rb") as output_file:
    final2 = pickle.load(output_file)

with open(OUT_PATH_3, "rb") as output_file:
    final3 = pickle.load(output_file)

with open(OUT_PATH_4, "rb") as output_file:
    final4 = pickle.load(output_file)

print (len(final1))
print (len(final2))
print (len(final3))
print (len(final4))

combined = final1 + final2 + final3 + final4

print("loaded everything into memory, damn")

combined = np.array([[int(x) for x in line] for line in combined])
print(combined.shape)

labels1 = [0] * len(final1)
labels2 = [1] * len(final2)
labels3 = [2] * len(final3)
labels4 = [3] * len(final4)

labels = np.array(labels1 + labels2 + labels3 + labels4)

print(labels.shape)

indices_shuffled = np.arange(len(labels))
np.random.shuffle(indices_shuffled)

labels = labels[indices_shuffled]
combined = combined[indices_shuffled]

col2 = np.roll(indices_shuffled, 100)
col3 = np.roll(indices_shuffled, 123)

out = []

for i in tqdm(range(len(combined))):
    line = [(combined[i], labels[i]), (combined[col2[i]], labels[col2[i]]), (combined[col3[i]], labels[col3[i]])]
    out.append(line)

with open(OUT, "wb") as output_file:
    pickle.dump(out, output_file)
    print("wrote output to", output_file)
