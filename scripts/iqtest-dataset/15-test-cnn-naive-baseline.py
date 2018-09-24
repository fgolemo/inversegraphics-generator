import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from hyperdash import Experiment

from inversegraphics_generator.img_dataset import IqImgDataset
from inversegraphics_generator.iqtest_objs import get_data_dir
from inversegraphics_generator.resnet50 import MultiResNet

BATCH = 32

exp = Experiment("[ig] cnn-naive")
exp.param("epoch", EPOCHS)
exp.param("batch", BATCH)
exp.param("learning rate", LEARNING_RATE)

ds = IqImgDataset("/data/lisa/data/iqtest/iqtest-dataset.h5", "test")
# ds = IqImgDataset(os.path.join(get_data_dir(), "test.h5"), "train/labeled")
dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)

model = MultiResNet().cuda()


model.load_state_dict(torch.load('model-e40-b32-lr0.0001.ckpt'))

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for question, answer in dl:
        answer = answer.cuda()
        outputs = model(question.cuda())
        _, predicted = torch.max(outputs.data, 1)
        total += answer.size(0)
        correct += (predicted == answer).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

exp.end()


