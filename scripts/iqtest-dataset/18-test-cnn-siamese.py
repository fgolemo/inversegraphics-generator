import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import os
from inversegraphics_generator.img_dataset import IqImgDataset
from inversegraphics_generator.iqtest_objs import get_data_dir
from inversegraphics_generator.resnet50 import MultiResNet

BATCH = 32

for size in [1000,10000]:
    for dsn in ["test","train/labeled"]:

        # ds = IqImgDataset("/lindata/datasets/ig/iqtest-dataset-ambient.h5", dsn)
        ds = IqImgDataset(os.path.join(get_data_dir(), "test.h5"), "train/labeled")
        dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)

        model = MultiResNet(siamese=True) #.cuda()

        model.load_state_dict(torch.load('model-siam2-s{}-e40-b64-lr0.0001.ckpt'.format(size)))

        # Test the model
        model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for question, answer in tqdm(dl):
                # answer = answer.cuda()

                print (answer.size())

                _, internals = model(question) #.cuda()
                print (internals[0].size())

                # find activation that is closest to ref
                dist1 = F.pairwise_distance(internals[0], internals[1], keepdim=True)
                print(dist1.size())
                dist2 = F.pairwise_distance(internals[0], internals[2], keepdim=True)
                dist3 = F.pairwise_distance(internals[0], internals[3], keepdim=True)

                dists = torch.stack((dist1,dist2,dist3))
                print (dists.size())

                min_dist = torch.argmin(dists, keepdim=True, dim=0).squeeze(0)

                predicted = torch.zeros((answer.size()[0],3)).long()

                for idx, line in enumerate(min_dist):
                    predicted[idx,line] = 1


                print (predicted)

                quit()

                _, ans = torch.max(answer.data, 1)
                total += answer.size(0)
                correct += (predicted == ans).sum().item()

            print('(Siamese) Accuracy of {} sample model on {}: {} %'.format(
                size,
                dsn,
                100 * correct / total
            ))

