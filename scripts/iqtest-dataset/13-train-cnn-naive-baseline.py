import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from hyperdash import Experiment

from inversegraphics_generator.img_dataset import IqImgDataset
from inversegraphics_generator.iqtest_objs import get_data_dir
from inversegraphics_generator.resnet50 import MultiResNet

EPOCHS = 5
BATCH = 5
LEARNING_RATE = 0.001

exp = Experiment("[ig] cnn-naive")
exp.param("epoch", EPOCHS)
exp.param("batch", BATCH)
exp.param("learning rate", LEARNING_RATE)

# ds = IqImgDataset("/data/lisa/data/iqtest/iqtest-dataset.h5", "train/labeled")
ds = IqImgDataset(os.path.join(get_data_dir(), "test.h5"), "train/labeled")
dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)

model = MultiResNet().cuda()
# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train the model
total_step = int(len(ds)/BATCH)
for epoch in range(EPOCHS):
    for i, (questions, answers) in enumerate(dl):
        # Forward pass
        outputs = model(questions.cuda())
        loss = criterion(outputs, answers.cuda())

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
    torch.save(model.state_dict(), 'model-e{}-b{}-lr{}.ckpt'.format(
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


