import shutil

import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from PIL import Image
from tqdm import tqdm

from torch.utils import data
from torchvision.transforms import ToTensor, Compose, Resize, ToPILImage

from config import config
from lib.module.PNSPlusNetwork import PNSNet as Network
from lib.dataloader.preprocess import *
from lib.dataloader.dataloader import get_video_dataset
from lib.module.EMA import EMA
import torch.nn as nn

import logging

class Normalize(object):
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, img):
        for i in range(3):
            img[:, :, i] -= float(self.mean[i])
        for i in range(3):
            img[:, :, i] /= float(self.std[i])
        return img

def cofficent_calculate(pred,gts,threshold=0.5):
    preds = pred > threshold
    intersection = (preds * gts).sum()
    union = preds.sum() + gts.sum()
    dice = 2.0 * intersection  / union
    iou = intersection/(union - intersection)
    return (dice, iou)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

parser = argparse.ArgumentParser()

if __name__ == '__main__':

    # seed = 18
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

    save_path = config.save_path_preds

    #curat folderul in care salvez predictiile
    #daria
    # Ensure save_path exists
    if os.path.exists(save_path):
        # Remove all contents of the directory
        shutil.rmtree(save_path)

    # Recreate empty directory
    os.makedirs(save_path, exist_ok=True)
    #end daria

    logging.basicConfig(filename=save_path + 'test.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')


    test_loader = get_video_dataset(config.test_split, test=True)
    test_loader = data.DataLoader(dataset=test_loader,
                                   batch_size=1,
                                   shuffle=False,
                                   num_workers=0,
                                   pin_memory=False,
                                   worker_init_fn=seed_worker
                                   #generator=torch.Generator().manual_seed(seed)
                                    )

    model_path = config.pth_path
    print('Load checkpoint:', model_path)
    logging.info('Load checkpoint: {}'.format(model_path))
    # model = Network(bn_out=(config.size[0] // 16, config.size[1] // 16), use_kan=config.use_kan).cuda()
    model = Network(bn_out=(config.size[0] // 8, config.size[1] // 8), use_kan=config.use_kan).cuda()
    model = nn.DataParallel(model)

    state_dict = torch.load(model_path, map_location=torch.device('cuda:0'))
    model.load_state_dict(state_dict['model_state_dict'])
    ema = EMA(model, decay=0.9998)
    if config.ema_test:
        ema.load(model, model_path)
    model.eval()

    reses = {}
    low_res = []
    dice_sum = 0.0
    size = 0
    last_print = 5
    batches = len(test_loader)
    if config.ema_test:
        ema.apply_shadow()
    with torch.no_grad():
        for batch_idx, (image, gt, img_pth) in enumerate(tqdm(test_loader)):
            #print(image.shape)
            image = image.cuda()
            gt = gt.cuda()
            # # contributie start
            # print("gt unique values:", torch.unique(gt))
            # # contributie end
            pred = model(image)
            for i in range(gt.shape[1]):
                dice = cofficent_calculate(pred[i], gt[0][i])[0]
                dice_sum += dice
                size += 1
            indexes = 1
            for i in range(gt.shape[1]):
                save_dir = save_path + "/" + img_pth[i+1][1][0].split('/')[-2]
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                img_save_path = save_dir + "/" + img_pth[i+1][1][0].split('/')[-1]
                # pil_img = ToPILImage()(pred[i]) -> original
                # contributie start
                pil_img = ToPILImage()((pred[i] > 0.5).float())
                # contributie end
                pil_img.save(img_save_path)

    if config.ema_test:
        ema.restore()
    
    meandice = dice_sum / size
    print('Mean Dice on frames:{:.4f}'.format(meandice))
    logging.info('Mean Dice on frames:{:.4f}'.format(meandice))
