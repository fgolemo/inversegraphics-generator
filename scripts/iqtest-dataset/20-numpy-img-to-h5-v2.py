import numpy as np
import h5py
from tqdm import tqdm

# from inversegraphics_generator.iqtest_objs import get_data_dir

# IN_PATH = get_data_dir() + "/{}000_baseline_data.npy"
IN_PATH = "/data/Florian/{}/{}000_baseline_data.npy"

# OUT_PATH = get_data_dir() + "/3diqtt-v2.h5"
OUT_PATH = "/data/3diqtt-v2-{}.h5"

CHUNK = 5000


# shape: 10000, 4, 128, 128, 3

# train: 100k, consisting of 90k unlabeled and 10k labeled
# test: 10k unlabeled
# val: 10k labeled


def prep(a):
    a = np.swapaxes(a, 2, 4)  # move color dim to back, but now r/g are swapped
    a = np.swapaxes(a, 2, 3)  # unswap r/g
    a /= 255  # assuming a.max() is around 255 rather than around 1
    a = np.clip(a, 0, 1)
    return a


with h5py.File(OUT_PATH.format("test"), "w") as f:
    test_q = f.create_dataset("questions", (10000, 4, 128, 128, 3), dtype=np.float32)

    for i in tqdm(range(2)):
        data = np.load(IN_PATH.format("test", (i + 1) * 5), mmap_mode="c")
        data = prep(data)

        col_a = data[:, 0, :, :, :]
        col_b = np.roll(data[:, 1, :, :, :], 1234, axis=0)
        col_c = np.roll(data[:, 2, :, :, :], 2468, axis=0)
        col_d = data[:, 3, :, :, :]  # copy of col_a

        cols = [col_b, col_c, col_d]

        questions = np.zeros((CHUNK, 4, 128, 128, 3), dtype=np.float32)
        answers = np.zeros((CHUNK), dtype=np.uint8)

        order = np.array([0, 1, 2])

        for idx in range(CHUNK):
            questions[idx, 0] = col_a[idx]
            np.random.shuffle(order)
            answers[idx] = np.where(order == 2)[0]
            questions[idx, 1] = cols[order[0]][idx]
            questions[idx, 2] = cols[order[1]][idx]
            questions[idx, 3] = cols[order[2]][idx]

        indices = np.arange(0, CHUNK)
        np.random.shuffle(indices)

        # mem-efficient shuffle in place
        rng = np.random.get_state()
        np.random.shuffle(questions)
        np.random.set_state(rng)
        np.random.shuffle(answers)

        test_q[(i * 5000):((i + 1) * 5000)] = questions
        # labeled_a[(i * 5000):((i + 1) * 5000)] = answers

        del data

print("test done, doing validation now")

with h5py.File(OUT_PATH.format("val"), "w") as f:
    val_q = f.create_dataset("questions", (10000, 4, 128, 128, 3), dtype=np.float32)
    val_a = f.create_dataset("answers", (10000,), dtype=np.uint8)

    for i in tqdm(range(2)):
        data = np.load(IN_PATH.format("train", (i + 21) * 5), mmap_mode="c")
        data = prep(data)

        col_a = data[:, 0, :, :, :]
        col_b = np.roll(data[:, 1, :, :, :], 2345, axis=0)
        col_c = np.roll(data[:, 2, :, :, :], 666, axis=0)
        col_d = data[:, 3, :, :, :]  # copy of col_a

        cols = [col_b, col_c, col_d]

        questions = np.zeros((CHUNK, 4, 128, 128, 3), dtype=np.float32)
        answers = np.zeros((CHUNK), dtype=np.uint8)

        order = np.array([0, 1, 2])

        for idx in range(CHUNK):
            questions[idx, 0] = col_a[idx]
            np.random.shuffle(order)
            answers[idx] = np.where(order == 2)[0]
            questions[idx, 1] = cols[order[0]][idx]
            questions[idx, 2] = cols[order[1]][idx]
            questions[idx, 3] = cols[order[2]][idx]

        indices = np.arange(0, CHUNK)
        np.random.shuffle(indices)

        # mem-efficient shuffle in place
        rng = np.random.get_state()
        np.random.shuffle(questions)
        np.random.set_state(rng)
        np.random.shuffle(answers)

        val_q[(i * 5000):((i + 1) * 5000)] = questions
        val_a[(i * 5000):((i + 1) * 5000)] = answers

        del data

print("val done, doing train now")

with h5py.File(OUT_PATH.format("train"), "w") as f:
    train_labeled_group = f.create_group("labeled")
    train_unlabeled_group = f.create_group("unlabeled")

    labeled_q = train_labeled_group.create_dataset("questions", (10000, 4, 128, 128, 3),
                                                   dtype=np.float32)
    labeled_a = train_labeled_group.create_dataset("answers", (10000,), dtype=np.uint8)
    unlabeled_q = train_unlabeled_group.create_dataset("questions", (90000, 4, 128, 128, 3), dtype=np.float32)

    ### TRAIN

    for i in tqdm(range(20)):
        data = np.load(IN_PATH.format("train", (i + 1) * 5), mmap_mode="c")
        data = prep(data)

        col_a = data[:, 0, :, :, :]
        col_b = np.roll(data[:, 1, :, :, :], 2345, axis=0)
        col_c = np.roll(data[:, 2, :, :, :], 666, axis=0)
        col_d = data[:, 3, :, :, :]  # copy of col_a

        cols = [col_b, col_c, col_d]

        questions = np.zeros((CHUNK, 4, 128, 128, 3), dtype=np.float32)
        answers = np.zeros((CHUNK), dtype=np.uint8)

        order = np.array([0, 1, 2])

        for idx in range(CHUNK):
            questions[idx, 0] = col_a[idx]
            np.random.shuffle(order)
            answers[idx] = np.where(order == 2)[0]
            questions[idx, 1] = cols[order[0]][idx]
            questions[idx, 2] = cols[order[1]][idx]
            questions[idx, 3] = cols[order[2]][idx]

        indices = np.arange(0, CHUNK)
        np.random.shuffle(indices)

        # mem-efficient shuffle in place
        rng = np.random.get_state()
        np.random.shuffle(questions)
        np.random.set_state(rng)
        np.random.shuffle(answers)

        labeled_q[(i * 5000):((i + 1) * 5000)] = questions
        labeled_a[(i * 5000):((i + 1) * 5000)] = answers

        del data
