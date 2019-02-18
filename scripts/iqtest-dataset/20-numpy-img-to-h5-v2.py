import numpy as np
import h5py
from tqdm import tqdm
# from inversegraphics_generator.iqtest_objs import get_data_dir

# IN_PATH = get_data_dir() + "/{}000_baseline_data.npy"
IN_PATH = "/network/data1/sai/Florian/train/{}000_baseline_data.npy"

# OUT_PATH = get_data_dir() + "/3diqtt-v2.h5"
OUT_PATH = "/network/data1/sai/Florian/train/3diqtt-v2.h5"

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



with h5py.File(OUT_PATH, "w") as f:
    train = f.create_group("train")
    test = f.create_group("test")
    val = f.create_group("val")

    train_labeled_group = train.create_group("labeled")
    train_unlabeled_group = train.create_group("unlabeled")

    labeled_q = train_labeled_group.create_dataset("questions", (10000, 4, 128, 128, 3),
                                                   dtype=np.float32,
                                                   compression="gzip")
    labeled_a = train_labeled_group.create_dataset("answers", (10000,), dtype=np.uint8, compression="gzip")

    unlabeled_q = train_unlabeled_group.create_dataset("questions", (90000, 4, 128, 128, 3), dtype=np.float32,
                                                       compression="gzip")

    test_q = test.create_dataset("questions", (10000, 4, 128, 128, 3), dtype=np.float32, compression="gzip")

    val_q = val.create_dataset("questions", (10000, 4, 128, 128, 3), dtype=np.float32, compression="gzip")
    val_a = val.create_dataset("answers", (10000,), dtype=np.uint8, compression="gzip")

    ### TRAIN

    for i in range(18):
        data = np.load(IN_PATH.format((i + 1) * 5), mmap_mode="c")
        data = prep(data)

        col_a = data[:, 0, :, :, :]
        col_b = np.roll(data[:, 1, :, :, :], 2345, axis=0)
        col_c = np.roll(data[:, 2, :, :, :], 666, axis=0)
        col_d = data[:, 3, :, :, :]  # copy of col_a

        cols = [col_b, col_c, col_d]

        questions = np.zeros((CHUNK, 4, 128, 128, 3), dtype=np.float32)
        answers = np.zeros((CHUNK), dtype=np.uint8)

        order = np.array([0, 1, 2])

        for idx in tqdm(range(CHUNK)):
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

        print ("slice done")

        del data


    # for i in range(18):
    #     data = load(IN_PATH.format((i + 1) * 5))
    #
    #     col_a = data[:, 0, :, :, :]
    #     col_b = np.roll(data[:, 1, :, :, :], 2345, axis=0)
    #     col_c = np.roll(data[:, 2, :, :, :], 666, axis=0)
    #     col_d = data[:, 3, :, :, :]  # copy of col_a
    #
    #     cols = [col_b, col_c, col_d]
    #
    #     questions = np.zeros((CHUNK, 4, 128, 128, 3), dtype=np.float32)
    #     answers = np.zeros((CHUNK), dtype=np.uint8)
    #
    #     order = np.array([0, 1, 2])
    #
    #     for idx in tqdm(range(CHUNK)):
    #         questions[idx, 0] = col_a[idx]
    #         np.random.shuffle(order)
    #         answers[idx] = np.where(order == 2)[0]
    #         questions[idx, 1] = cols[order[0]][idx]
    #         questions[idx, 2] = cols[order[1]][idx]
    #         questions[idx, 3] = cols[order[2]][idx]
    #
    #     indices = np.arange(0, CHUNK)
    #     np.random.shuffle(indices)
    #
    #     # mem-efficient shuffle in place
    #     rng = np.random.get_state()
    #     np.random.shuffle(questions)
    #     np.random.set_state(rng)
    #     np.random.shuffle(answers)
    #
    #     labeled_q[(i * 5000):((i + 1) * 5000)] = questions
    #     labeled_a[(i * 5000):((i + 1) * 5000)] = answers

