import argparse
import os
import sys

import Training
import pathspec
import torch


def cp_projects(to_path):
    with open('./.gitignore', 'r') as fp:
        ign = fp.read()
    ign += '\n.git'
    spec = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, ign.splitlines())
    all_files = {os.path.join(root, name) for root, dirs, files in os.walk('./') for name in files}
    matches = spec.match_files(all_files)
    matches = set(matches)
    to_cp_files = all_files - matches

    for f in to_cp_files:
        dirs = os.path.join(to_path, 'code', os.path.split(f[2:])[0])
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        os.system('cp %s %s' % (f, os.path.join(to_path, 'code', f[2:])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_path', default='/4T/wenhu/eccv/open_source_res_256', type=str, help='path for output')
    parser.add_argument('--Training', default=False, type=bool, help='Training or not')
    parser.add_argument('--init_method', default='tcp://127.0.0.1:32222', type=str, help='init_method')
    parser.add_argument('--data_root', default='/4T/wenhu/dataset/RGB-SOD/', type=str, help='data path')
    parser.add_argument('--dt_root', default='/camsal_two/', type=str, help='DT data path')
    parser.add_argument('--test_paths', type=str, default='DUTS/DUTS-TE+ECSSD+HKU-IS+PASCAL-S+DUT-O')
    parser.add_argument('--img_size', default=256, type=int, help='network input size')
    parser.add_argument('--test_size', type=int, default=256, help='testing size')

    parser.add_argument('--pretrained_model', default='', type=str, help='load Pretrained model')
    parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='epochs')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--batch_size', default=16, type=int, help='batch_size')
    parser.add_argument('--query_num', default=3, type=int, help='query-num')

    parser.add_argument('--aux_loss', default=1.0, type=float, help='batch_size')

    parser.add_argument('--trainset', default='DUTS/DUTS-TR/', type=str, help='Trainging set')
    parser.add_argument('--methods', type=str, default='Baseline', help='evaluated method name')

    args = parser.parse_args()

    args.save_model_dir = args.save_path + '/checkpoint/'
    args.save_dir = args.save_path
    args.save_test_path_root = args.save_path + '/preds/'

    num_gpus = torch.cuda.device_count()
    if args.Training:
        if os.path.exists(args.save_path):
            print("save_path exits! Training Error")
            sys.exit()
        cp_projects(args.save_dir)
        Training.train_net(num_gpus=num_gpus, args=args)
