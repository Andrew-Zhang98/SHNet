import os
import random
from datetime import datetime

import Testing
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from Evaluation import main as eval_main
from Models.SaliencyFormer import SaliencyFormer
from newdataset_cam import get_train_loader
from tensorboardX import SummaryWriter
from torch import optim
from torch.autograd import Variable

step = 0
best_mae = 1
best_epoch = 0


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, max_epoch=150):
    for param_group in optimizer.param_groups:
        optimizer.param_groups[0]['lr'] = (1 - abs((epoch + 1) / (max_epoch + 1) * 2 - 1)) * init_lr * 0.1
        optimizer.param_groups[1]['lr'] = (1 - abs((epoch + 1) / (max_epoch + 1) * 2 - 1)) * init_lr
        lr = param_group['lr']
    return lr


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def seed_torch():
    # seed = int(time.time()*256) % (2**32-1)
    seed = 666
    # 保存随机种子
    print("~~~random_seed:", seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def train_net(num_gpus, args):
    mp.spawn(main, nprocs=num_gpus, args=(num_gpus, args))


def main(local_rank, num_gpus, args):
    print(">>>>>>>")
    seed_torch()

    cudnn.benchmark = True
    writer = SummaryWriter(args.save_dir + '/logging')

    dist.init_process_group(backend='nccl', init_method=args.init_method, world_size=num_gpus, rank=local_rank)

    torch.cuda.set_device(local_rank)

    model = SaliencyFormer()

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print("all_model_parameters: ", num_params / 1e6)

    model.train()
    model.cuda()

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True)

    KD = nn.MSELoss().cuda()

    preprocess = None

    base, head = [], []
    for name, param in model.named_parameters():
        if 'resnet' in name:
            base.append(param)
        else:
            head.append(param)

    optimizer = optim.Adam([{'params': base, 'lr': args.lr}, {'params': head, 'lr': args.lr}])

    image_root = args.data_root + args.trainset + '/DUTS-TR-Image/'
    gt_root = args.data_root + args.trainset + '/DUTS-TR-Mask/'
    contour_root = args.data_root + args.trainset + '/DUTS-TR-Contour/'
    dt_root = args.data_root + args.trainset + args.dt_root

    train_loader = get_train_loader(image_root, gt_root, contour_root, dt_root, batchsize=args.batch_size,
                                    trainsize=args.img_size, preprocess=preprocess, num_gpus=num_gpus, rank=local_rank,
                                    query_num=args.query_num)

    if (local_rank == 0):
        print('''
            Starting training:
                Train epochs: {}
                Batch size: {}
                Learning rate: {}
                Training size: {}
                Image size: {},{}
                DT root: {}
                aux_loss: {}
                query_num: {}
            '''.format(args.epochs, args.batch_size, args.lr, len(train_loader.dataset), args.img_size, args.test_size,
                       args.dt_root, args.aux_loss, args.query_num))

    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    best_res = {}
    for dataset_name in args.test_paths.split('+'):
        best_res[dataset_name] = {'MAE': 1.0, "Sm": 0.0, "MeanFm": 0.0}
    for epoch in range(1, args.epochs + 1):
        cur_lr = adjust_lr(optimizer, args.lr, epoch, args.epochs)

        if (local_rank == 0):
            print('Starting epoch {}/{}.'.format(epoch, args.epochs))
            print('epoch:{0}-------lr:{1}, {2}'.format(epoch, optimizer.param_groups[0]['lr'],
                                                       optimizer.param_groups[1]['lr']))

        train_epoch(train_loader, model, KD, optimizer, epoch, local_rank, writer, args)

        if (local_rank == 0) and (epoch % 5 == 0 or epoch >= int(args.epochs * 0.7) or epoch == 1):
            Testing.test_net(args, epoch)
            epoch_res = eval_main.evaluate(args, epoch)
            for dataset_name in args.test_paths.split('+'):
                if epoch_res[dataset_name]['MAE'] < best_res[dataset_name]['MAE']:
                    best_res[dataset_name]['MAE'] = epoch_res[dataset_name]['MAE']
                if epoch_res[dataset_name]['Sm'] > best_res[dataset_name]['Sm']:
                    best_res[dataset_name]['Sm'] = epoch_res[dataset_name]['Sm']
                if epoch_res[dataset_name]['MeanFm'] > best_res[dataset_name]['MeanFm']:
                    best_res[dataset_name]['MeanFm'] = epoch_res[dataset_name]['MeanFm']

                writer.add_scalar('eval/' + dataset_name.split('/')[-1] + '_MAE', best_res[dataset_name]['MAE'], epoch)
                writer.add_scalar('eval/' + dataset_name.split('/')[-1] + '_Sm', best_res[dataset_name]['Sm'], epoch)
                writer.add_scalar('eval/' + dataset_name.split('/')[-1] + '_MeanFm', best_res[dataset_name]['MeanFm'],
                                  epoch)
        print("epoch best results:", best_res)


def train_epoch(train_loader, model, KD, optimizer, epoch, local_rank, writer, args):
    save_path = args.save_model_dir
    total_step = len(train_loader)

    global step
    model.train()
    loss_all_sum = 0.
    loss_sal_sum = 0.
    loss_con_sum = 0.
    epoch_step = 0
    try:
        for i, data_batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images, gts, contours, dt_masks = data_batch
            images = images.cuda()
            gts = gts.cuda()
            contours = contours.cuda()
            dt_masks = dt_masks.cuda()

            images, gts, contours, dt_masks = Variable(images.cuda(local_rank, non_blocking=True)), \
                                              Variable(gts.cuda(local_rank, non_blocking=True)), \
                                              Variable(contours.cuda(local_rank, non_blocking=True)), \
                                              Variable(dt_masks.cuda(local_rank, non_blocking=True))

            mask, contour, mid_pre, _ = model(images)
            pred_mask, pred_masks = mask
            pred_contour, pred_contours = contour

            # structure_loss
            loss_sal = structure_loss(pred_mask, gts)
            loss_sal += structure_loss(mid_pre[0], gts) * 0.5
            loss_sal += structure_loss(mid_pre[1], gts) * 0.5
            loss_sal += structure_loss(mid_pre[2], gts) * 0.8
            loss_sal += structure_loss(mid_pre[3], gts) * 0.8

            # hierarchy_loss
            loss_sal_aux = loss_sal * 0.
            for m in pred_masks:
                map_i = m
                mask_i = F.interpolate(dt_masks, size=[m.shape[-2], m.shape[-1]], mode="bilinear")
                loss_sal_aux += KD(map_i, mask_i)
            loss_sal_aux = loss_sal_aux * args.aux_loss

            loss_s = loss_sal + loss_sal_aux

            loss = loss_s

            loss.backward()

            del images, gts, contours, dt_masks, mask, contour, mid_pre, pred_mask, pred_masks
            torch.cuda.empty_cache()

            clip_gradient(optimizer, args.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all_sum += loss.cpu().data.item()
            loss_sal_sum += loss_sal.cpu().data.item()

            if (local_rank == 0) and (i % 50 == 0 or i == total_step or i == 1):
                print(
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss_sal: {:.4f} Loss_sal_aux: {:0.4f}'.
                        format(datetime.now(), epoch, args.epochs, i, total_step, loss_sal.data, loss_sal_aux.data))

                whole_iter_num = (epoch - 1) * total_step + i
                writer.add_scalar('train/LR', optimizer.param_groups[1]['lr'], whole_iter_num + 1)
                writer.add_scalar('train/total_loss', loss_all_sum / epoch_step, whole_iter_num + 1)
                writer.add_scalar('train/saliency_loss', loss_sal_sum / epoch_step, whole_iter_num + 1)
                writer.add_scalar('train/contour_loss', loss_con_sum / epoch_step, whole_iter_num + 1)

        if (local_rank == 0):
            print('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_all_AVG: {:.4f}, Loss_saliency_AVG: {:.4f}'.format(epoch,
                                                                                                          args.epochs,
                                                                                                          loss_all_sum / epoch_step,
                                                                                                          loss_sal_sum / epoch_step))

        if (local_rank == 0) and (epoch % 5 == 0 or epoch >= int(args.epochs * 0.7) or epoch == 1):
            torch.save(model.state_dict(), save_path + '/Model_epoch_{}.pth'.format(epoch))

    except KeyboardInterrupt:
        if (local_rank == 0):
            print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if (local_rank == 0):
            torch.save(model.state_dict(), save_path + '/Interrupt_Model_epoch_{}.pth'.format(epoch))
            print('save checkpoints successfully!')
        raise
