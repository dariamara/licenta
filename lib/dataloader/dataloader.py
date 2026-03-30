import os

import torch
from torch.utils.data import Dataset

from lib.dataloader.preprocess import *
from scripts.config import config


class VideoDataset(Dataset):
    def __init__(self, video_dataset, augment, transform=None, time_interval=1, test=False):
        super(VideoDataset, self).__init__()
        self.time_clips = config.video_time_clips
        self.test = test
        self.video_train_list = []
        self.augment = augment

        video_root = os.path.join(config.dataset_root, video_dataset)
        img_root = os.path.join(video_root, 'Frame')
        gt_root = os.path.join(video_root, 'GT')

        cls_list = os.listdir(img_root)
        self.video_filelist = {}
        for cls in cls_list:
            if cls[-1].isdigit():
                self.video_filelist[cls] = []

                cls_img_path = os.path.join(img_root, cls)
                cls_label_path = os.path.join(gt_root, cls)

                tmp_list = os.listdir(cls_img_path)
                tmp_list.sort(key=lambda name: (
                    int(name.split('-')[0].split('_')[-1]),
                    int(name.split('_a')[1].split('_')[0]),
                    int(name.split('_image')[1].split('.jpg')[0])))

                for filename in tmp_list:
                    self.video_filelist[cls].append((
                        os.path.join(cls_img_path, filename),
                        os.path.join(cls_label_path, filename.replace(".jpg", ".png"))
                    ))
        # ensemble
        for cls in cls_list:
            if cls[-1].isdigit():
                li = self.video_filelist[cls]
                if self.test:
                    begin = 1  # change for inference from first frame
                    while begin < len(li):
                        if len(li) - begin - 1 < self.time_clips:
                            begin = len(li) - self.time_clips
                        batch_clips = []
                        batch_clips.append(li[0])
                        for t in range(self.time_clips):
                            batch_clips.append(li[begin + time_interval * t])
                        begin += self.time_clips
                        self.video_train_list.append(batch_clips)
                else:
                    for begin in range(1, len(li) - (self.time_clips - 1) * time_interval - 1):
                        batch_clips = []
                        batch_clips.append(li[0])
                        for t in range(self.time_clips):
                            batch_clips.append(li[begin + time_interval * t])
                        self.video_train_list.append(batch_clips)
        self.img_label_transform = transform

    def __getitem__(self, idx):
        img_label_li = self.video_train_list[idx]
        index = idx
        img_li = []
        label_li = []
        for idx, (img_path, label_path) in enumerate(img_label_li):
            img = Image.open(img_path).convert('RGB')
            label = Image.open(label_path).convert('L')
            img_li.append(img)
            label_li.append(label)
        if self.augment:
            img_li, label_li = cv_random_flip(img_li, label_li)
            img_li, label_li = randomRotation(img_li, label_li)
            img_li = colorEnhance(img_li)
            for i in range(len(label_li)):
                label_li[i] = randomPeper(label_li[i])
        img_li, label_li = self.img_label_transform(img_li, label_li)
        #daria
        # T = len(img_li)
        # H, W = img_li[0].shape[-2], img_li[0].shape[-1]
        # IMG = torch.zeros(T, 3, H, W)
        # LABEL = torch.zeros(T - 1, 1, H, W)
        #end
        for idx, (img, label) in enumerate(zip(img_li, label_li)):
            #original -> fara asta
            # if img.shape[0] == 1:
            #     img = img.repeat(3, 1, 1)
            #
            #print(f"DEBUG: Frame {idx}, Label Shape: {label.shape}, Label Ndim: {label.ndim}")
            if label.ndim == 2:
                label = label.unsqueeze(0)

            #print(f"DEBUG: Frame {idx}, Label Shape: {label.shape}, Label Ndim: {label.ndim}")
            #end

            if idx == 0:
                #original
                IMG = torch.zeros(len(img_li), *(img.shape))
                LABEL = torch.zeros(len(img_li) - 1, *(label.shape))
                #end original
                #daria
                #IMG = torch.zeros(len(img_li), 3, img.shape[1], img.shape[2])
                #LABEL = torch.zeros(len(img_li) - 1, 1, label.shape[1], label.shape[2])
                #end daria
                IMG[idx, :, :, :] = img
            else:
                IMG[idx, :, :, :] = img
                LABEL[idx - 1, :, :, :] = label
        if self.test != True:
            return IMG, LABEL
        else:
            return IMG, LABEL, img_label_li

    def __len__(self):
        return len(self.video_train_list)


def get_video_dataset(split, augment=False, test=False):
    statistics = torch.load(config.data_statistics)
    trsf_main = Compose_imglabel([
        Resize_video(config.size[0], config.size[1]),
        toTensor_video(),
        Normalize_video(statistics["mean"], statistics["std"])
    ])
    train_loader = VideoDataset(split, augment=augment, transform=trsf_main, time_interval=1, test=test)

    return train_loader


if __name__ == "__main__":
    statistics = torch.load(config.data_statistics)
    trsf_main = Compose_imglabel([
        Resize_video(config.size[0], config.size[1]),
        toTensor_video(),
    ])
    train_loader = VideoDataset(config.train_split, transform=trsf_main, time_interval=1)
