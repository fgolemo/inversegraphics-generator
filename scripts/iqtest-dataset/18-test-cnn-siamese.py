import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from inversegraphics_generator.img_dataset import IqImgDataset
from inversegraphics_generator.iqtest_objs import get_data_dir
from inversegraphics_generator.resnet50 import MultiResNet

BATCH = 32

for size in [1000,10000]:
    for dsn in ["test","train/labeled"]:

        ds = IqImgDataset("/data/lisa/data/iqtest/iqtest-dataset-ambient.h5", dsn)
        # ds = IqImgDataset(os.path.join(get_data_dir(), "test.h5"), "train/labeled")
        dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)

        model = MultiResNet().cuda()

        model.load_state_dict(torch.load('model-siam-s{}-e40-b32-lr0.0001.ckpt'.format(size)))

        # Test the model
        model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for question, answer in tqdm(dl):
                answer = answer.cuda()
                outputs = model(question.cuda())
                _, ans = torch.max(answer.data, 1)
                _, predicted = torch.max(outputs.data, 1)
                total += answer.size(0)
                correct += (predicted == ans).sum().item()

            print('(Siamese) Accuracy of {} sample model on {}: {} %'.format(
                size,
                dsn,
                100 * correct / total
            ))

