import os
import pickle
import numpy as np
from tqdm import tqdm

# OUT = os.path.join(get_data_dir(), "aggregated.pkl")
IN_PATH = os.path.expanduser("~/data/ig/aggregated.pkl")
OUT_PATH = os.path.expanduser("~/data/ig/aggregated-question.pkl")
PARAMS = [(11, 5), (13, 5), (14, 6), (15, 8)]

with open(IN_PATH, "rb") as output_file:
    aggr = pickle.load(output_file)

order = np.arange(4)

print(len(aggr))

# og4 = ObjGenerator(GRID_SIZE_4, 1.0)

refs = []
cola = []
colb = []
colc = []
cold = []
answ = []

for i in tqdm(range(len(aggr))):

    # take out reference answer
    refs.append(aggr[i][0])

    line_idx = order.copy()

    # randomize order of answers in this line
    np.random.shuffle(line_idx)

    line_tmp = np.array(aggr[i])[line_idx]
    cola.append((line_tmp[0][0],line_tmp[0][1]))
    colb.append((line_tmp[1][0],line_tmp[1][1]))
    colc.append((line_tmp[2][0],line_tmp[2][1]))
    cold.append((line_tmp[3][0],line_tmp[3][1]))

    right_solution_idx = np.where(line_idx == 0)[0][0]
    answ.append(right_solution_idx)


with open(OUT_PATH, "wb") as output_file:
    pickle.dump(refs, output_file)
    pickle.dump(cola, output_file)
    pickle.dump(colb, output_file)
    pickle.dump(colc, output_file)
    pickle.dump(cold, output_file)
    pickle.dump(answ, output_file)
    print ("wrote output to",output_file)

