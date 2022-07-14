import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from Models.SaliencyFormer import SaliencyFormer
from newdataset_cam import test_dataset
import argparse


def test_net(args, test_epoch):
    cudnn.benchmark = True

    data_root = args.data_root
    save_test_path_root = args.save_test_path_root

    net = SaliencyFormer()

    net.cuda()
    net.eval()

    # load model (multi-gpu)
    model_path = args.save_model_dir + '/Model_epoch_{}.pth'.format(test_epoch)
    state_dict = torch.load(model_path)
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    net.load_state_dict(new_state_dict)
    print('Model loaded from {}'.format(model_path))

    test_paths = args.test_paths
    test_datasets = test_paths.split('+')

    for dataset in test_datasets:
        print('Testing on ', dataset)
        save_path = save_test_path_root + '/' + dataset + '/Baseline/'
        save_path_aux = save_test_path_root + '/' + dataset + '_aux' + '/Baseline/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(save_path_aux):
            os.makedirs(save_path_aux)
        image_root = data_root + dataset + '/images/'
        gt_root = data_root + dataset + '/GT/'
        test_loader = test_dataset(image_root, gt_root, args.img_size)
        for i in range(test_loader.size):
            image, gt, name, image_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            with torch.no_grad():
                mask, contour, _, _ = net(image)
                res = mask[0]
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            cv2.imwrite(save_path + name, res * 255)
        print('Test Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', default='/4T/wenhu/dataset/RGB-SOD/', type=str, help='data path')
    parser.add_argument('--img_size', default=256, type=int, help='network input size')
    parser.add_argument('--test_paths', type=str, default='DUTS/DUTS-TE+ECSSD+HKU-IS+PASCAL-S+DUT-O')

    args = parser.parse_args()

    args.save_model_dir = args.save_path + '/checkpoint/'
    args.save_test_path_root = args.save_path + '/preds/'

    test_net(args, 100)

