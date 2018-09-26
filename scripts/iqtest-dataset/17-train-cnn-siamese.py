import os

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from hyperdash import Experiment

from inversegraphics_generator.img_dataset import IqImgDataset
from inversegraphics_generator.iqtest_objs import get_data_dir
from inversegraphics_generator.resnet50 import MultiResNet, ContrastiveLoss

EPOCHS = 40
BATCH = 32
LEARNING_RATE = 0.0001
SIZE = 1000

exp = Experiment("[ig] cnn-siamese")
exp.param("epoch", EPOCHS)
exp.param("size", SIZE)
exp.param("batch", BATCH)
exp.param("learning rate", LEARNING_RATE)

# ds = IqImgDataset("/data/lisa/data/iqtest/iqtest-dataset-ambient.h5", "train/labeled", max_size=SIZE)
ds = IqImgDataset(os.path.join(get_data_dir(), "test.h5"), "train/labeled", max_size=SIZE)
dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)

model = MultiResNet(siamese=True).cuda()
# Loss and optimizer
criterion_basic = nn.BCELoss()
criterion_contrast = ContrastiveLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train the model
total_step = int(len(ds)/BATCH)
for epoch in range(EPOCHS):
    for i, (questions, answers) in enumerate(dl):
        # Forward pass
        outputs, internals = model(questions.cuda())

        loss = criterion_basic(outputs, answers.cuda())

        for b in range(min(BATCH,len(answers))):
            # get matching ref-answer loss
            other_answers = [0,1,2]

            correct_ans_idx = np.where(answers[b] == 1)[0][0]
            other_answers.remove(correct_ans_idx)

            loss += criterion_contrast(internals[0], internals[1+correct_ans_idx], 0)

            # add contrastive loss for the non-matching answers
            loss += criterion_contrast(internals[0], internals[1+other_answers[0]], 1)
            loss += criterion_contrast(internals[0], internals[1+other_answers[1]], 1)


        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        exp.metric("epoch", epoch)
        exp.metric("loss", loss.item())

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, EPOCHS, i + 1, total_step, loss.item()))

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model-siam-s{}-e{}-b{}-lr{}.ckpt'.format(
        SIZE,
        EPOCHS,
        BATCH,
        LEARNING_RATE))

# # Test the model
# model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
#     print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

exp.end()


