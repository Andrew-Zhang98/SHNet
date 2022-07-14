import os
import random

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageEnhance


def cv_random_flip(img, label, depth, dt):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        dt = dt.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, depth, dt


def randomCrop(image, label, depth, dt):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region), depth.crop(random_region), dt.crop(random_region)


def randomRotation(image, label, depth, dt):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        depth = depth.rotate(random_angle, mode)
        dt = dt.rotate(random_angle, mode)
    return image, label, depth, dt


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


# dataset for training
class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, contour_root, dt_root, trainsize, preprocess, query_num):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')
                    or f.endswith('.jpg')]
        self.contours = [contour_root + f for f in os.listdir(contour_root) if f.endswith('.png')
                         or f.endswith('.jpg')]
        self.dt = [dt_root + f for f in os.listdir(dt_root) if f.endswith('.png')
                   or f.endswith('.jpg')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.contours = sorted(self.contours)
        self.dt = sorted(self.dt)

        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.N = 6

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        contour = self.binary_loader(self.contours[index])
        dt = self.binary_loader(self.dt[index])

        image, gt, contour, dt = cv_random_flip(image, gt, contour, dt)
        image, gt, contour, dt = randomCrop(image, gt, contour, dt)
        image, gt, contour, dt = randomRotation(image, gt, contour, dt)

        masks = self.camsal_split_1(dt, gt)
        image = colorEnhance(image)
        image = self.img_transform(image)

        gt = self.gt_transform(gt)

        masks = [self.gt_transform(mask_i) for mask_i in masks]
        masks = torch.cat(masks, dim=0)
        contour = self.gt_transform(contour)

        return image, gt, contour, masks

    def dt_split(self, dt, gt):
        N = self.N
        overlap = 0.05
        dt, gt = np.array(dt) * 1.0, np.array(gt) * 1.0
        dt = dt / 255.0
        masks = []
        for i in range(N):
            mask_i = np.logical_and(dt >= i / N - overlap, dt <= (1 + i) / N + overlap)
            mask_i = mask_i * 1.0 * gt
            mask_i.dtype = 'uint8'
            masks.append(Image.fromarray(mask_i))
        return masks

    def camsal_split_1(self, dt, gt):
        dt, gt = np.array(dt) * 1.0, np.array(gt) * 1.0
        gt = cv2.resize(gt, (256, 256))
        dt = cv2.resize(dt, (256, 256))
        N = self.N
        masks = []
        index = [0, 4]
        last_i = 0

        total_mask = np.zeros_like(dt)

        for i in range(N):
            if i not in index:
                continue

            logical_and = np.logical_and(dt > last_i, dt <= i + 1)
            mask_i = np.where(logical_and, 1, 0)
            mask_i = mask_i * 1.0 * gt
            mask_i = np.uint8(mask_i) * 255

            total_mask = np.logical_or(total_mask, mask_i)

            masks.append(Image.fromarray(mask_i).convert('L'))
            last_i = i + 1

        mask_i = np.where(total_mask > 0, 0, 1)
        mask_i = np.uint8(mask_i)
        masks.append(Image.fromarray(mask_i).convert('L'))
        return masks

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.contours)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


# dataloader for training
def get_train_loader(image_root, gt_root, contour_root, dt_root, batchsize, trainsize, num_gpus, rank, query_num,
                     preprocess=None, shuffle=True, num_workers=6, pin_memory=True):
    train_dataset = SalObjDataset(image_root, gt_root, contour_root, dt_root, trainsize, preprocess, query_num)

    sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=num_gpus,
        shuffle=shuffle,
        rank=rank,
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, num_workers=num_workers,
                                               pin_memory=True,
                                               sampler=sampler,
                                               drop_last=True,
                                               )
    return train_loader


# test dataset and loader
class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        image_for_post = self.rgb_loader(self.images[self.index])
        image_for_post = image_for_post.resize(gt.size)
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image, gt, name, np.array(image_for_post)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size
