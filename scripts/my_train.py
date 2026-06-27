import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

from config import config
from lib.dataloader.dataloader import get_video_dataset
from lib.module.EMA import EMA
from lib.module.PNSPlusNetwork import PNSNet as Network
from lib.utils.utils import cosine_scheduler, adjust_lr_step


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, *inputs):
        pred, target = tuple(inputs)
        total_loss = F.binary_cross_entropy(pred.squeeze(), target.squeeze().float())
        return total_loss

class MixedLoss(nn.Module):
    def __init__(self):
        super(MixedLoss, self).__init__()

    def forward(self, *inputs):
        pred, target = tuple(inputs)
        pred = pred.squeeze()
        target = target.squeeze()
        cross_entropy_loss = F.binary_cross_entropy(pred, target.float())
        eps = 1e-14
        preds = pred > 0.5
        intersection = (preds * target).sum()
        union = preds.sum() + target.sum()
        dice_loss = (1 - (2.0 * intersection + eps)  / (union + eps))
        return cross_entropy_loss + dice_loss

def cofficent_calculate(pred,gts,threshold=0.5):
    eps = 1e-14
    preds = pred > threshold
    intersection = (preds * gts).sum()
    union = preds.sum() + gts.sum()
    dice = (2.0 * intersection + eps) / (union + eps)
    iou = (intersection + eps) /(union - intersection + eps)
    return (dice, iou)

def train(train_loader, model, optimizer, epoch, save_path, loss_func):
    global step
    model.cuda().train()
    loss_all = 0
    epoch_step = 0
    dice_sum = 0.0
    size = 0

    try:
        for i, (images, gts) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            lr_idx = i + epoch * total_step - 1

            adjust_lr_step(optimizer, schedule_backbone[lr_idx], 'backbone_params')
            adjust_lr_step(optimizer, schedule_head[lr_idx], 'head_params')

            #print(images.shape)

            images = images.cuda()
            gts = gts.cuda()
            
            preds = model(images)
            
            loss = loss_func(preds.squeeze().contiguous(), gts.contiguous().view(-1, *(gts.shape[2:])))
            loss.backward()

            eval_preds = preds.contiguous().view(*(gts.shape))

            for batch in range(gts.shape[0]):
                for j in range(gts.shape[1]):
                    dice = cofficent_calculate(eval_preds[batch][j], gts[batch][j])[0]
                    dice_sum += dice
                    size += 1

            #clip_gradient(optimizer, config.clip)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.clip)

            # backbone_norm = 0
            # head_norm = 0
            
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         if name.startswith("module.feature_extractor"):
            #             param_norm = param.grad.data.norm(2)
            #             backbone_norm += param_norm.item() ** 2
            #         else:
            #             param_norm = param.grad.data.norm(2)
            #             head_norm += param_norm.item() ** 2
            
            # backbone_norm = backbone_norm ** 0.5
            # head_norm = head_norm ** 0.5
            # backbone_grad_norms.append(backbone_norm)
            # head_grad_norms.append(head_norm)

            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 20 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.
                      format(datetime.now(), epoch, config.epoches, i, total_step, loss.data))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.
                    format(epoch, config.epoches, i, total_step, loss.data))
                cur_lr = optimizer.param_groups[1]['lr']
                back_lr = optimizer.param_groups[0]['lr']

                print('Lr: {:.8f}'.format(cur_lr))
                logging.info('Lr: {:.8f}'.format(cur_lr))

                print('Lr: {:.8f}'.format(back_lr))
                logging.info('Lr: {:.8f}'.format(back_lr))
                if use_ema:
                    ema.update()
                
        # b_avg = sum(backbone_grad_norms) / len(backbone_grad_norms)
        # b_max = max(backbone_grad_norms)
        # b_p95 = sorted(backbone_grad_norms)[int(0.95*len(backbone_grad_norms))]
        
        # h_avg = sum(head_grad_norms) / len(head_grad_norms)
        # h_max = max(head_grad_norms)
        # h_p95 = sorted(head_grad_norms)[int(0.95*len(head_grad_norms))]
        
        # print('Epoch gradient stats - Backbone: Avg={:.4f}, Max={:.4f}, P95={:.4f}'.format(b_avg, b_max, b_p95))
        # print('Epoch gradient stats - Head: Avg={:.4f}, Max={:.4f}, P95={:.4f}'.format(h_avg, h_max, h_p95))
        # logging.info('Epoch gradient stats - Backbone: Avg={:.4f}, Max={:.4f}, P95={:.4f}'.format(b_avg, b_max, b_p95))
        # logging.info('Epoch gradient stats - Head: Avg={:.4f}, Max={:.4f}, P95={:.4f}'.format(h_avg, h_max, h_p95))

        os.makedirs(os.path.join(save_path, "epoch_%d" % (epoch + 1)), exist_ok=True)
        save_root = os.path.join(save_path, "epoch_%d" % (epoch + 1))
        torch.save(
            {'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'model_state_dict': model.state_dict(),
            'ema_state_dict': ema.shadow,
            },
              os.path.join(save_root, "PNSPlus.pth"))

        loss_all /= epoch_step
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, config.epoches, loss_all))
        meandice = dice_sum / size
        print('{} [Train Info]: Epoch [{:03d}/{:03d}], Dice: {:.4f}'.format(datetime.now(), epoch, config.epoches, meandice))
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Dice: {:.4f}'.format(epoch, config.epoches, meandice))

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save({'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'model_state_dict': model.state_dict(),
            'ema_state_dict': ema.shadow,
            },
                    save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise

def val(val_loader, model, epoch, loss_func):
    model.cuda().eval()
    dice_sum = 0.0
    total_batches = len(val_loader)
    size = 0
    loss_all = 0
    epoch_step = 0
    if use_ema:
        ema.apply_shadow()
    with torch.no_grad():
        for i, (images, gts) in enumerate(val_loader):

            images = images.cuda()
            gts = gts.cuda()
            
            preds = model(images)

            for j in range(gts.shape[1]):
                dice = cofficent_calculate(preds[j], gts[0][j])[0]
                dice_sum += dice
                size += 1
            
            loss = loss_func(preds.contiguous(), gts.contiguous())
            loss_all += loss.data
            epoch_step += 1

    if use_ema:
        ema.restore()

    loss_all /= epoch_step
    logging.info('[Val Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, config.epoches, loss_all))
    meandice = dice_sum / size
    print('{} [Val Info]: Epoch [{:03d}/{:03d}], Dice: {:.4f}'.format(datetime.now(), epoch, config.epoches, meandice))
    logging.info('[Val Info]: Epoch [{:03d}/{:03d}], Dice: {:.4f}'.format(epoch, config.epoches, meandice))

def load_checkpoint(model, optimizer, filename='checkpoint.pth.tar'):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Loaded checkpoint from '{filename}'. Resuming from epoch {start_epoch}.")
    return start_epoch

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

if __name__ == '__main__':

    # seed = 18
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

    # 3contributie
    model = Network(bn_out=(config.size[0] // 16, config.size[1] // 16), use_kan=config.use_kan, use_maxvit=config.use_maxvit).cuda()
    # 3contributie
    model = nn.DataParallel(model)

    cudnn.benchmark = True

    ema = EMA(model, decay=0.9992)
    if config.checkpoint_path != '':
        ema.load(model, config.checkpoint_path)
    use_ema = config.ema_train

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if name.startswith("module.feature_extractor"):
            backbone_params.append(param)
        else:
            head_params.append(param)

    print('Nr. backbone params: {:03d}'.format(len(backbone_params)))
    print('Nr. head params: {:03d}'.format(len(head_params)))

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': config.backbone_lr, 'weight_decay': config.backbone_weight_decay, 'name': 'backbone_params'},
        {'params': head_params, 'lr': config.head_lr, 'weight_decay': config.head_weight_decay, 'name': 'head_params'}])
    
    save_path = config.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    loss_func = MixedLoss()

    loss_func_val = MixedLoss()

    # load data
    print('load data...')
    train_loader = get_video_dataset(config.train_split, config.augment)
    train_loader = data.DataLoader(dataset=train_loader,
                                   batch_size=config.batchsize,
                                   shuffle=True,
                                   num_workers=4, #initial era 4
                                   pin_memory=False,
                                   worker_init_fn=seed_worker
                                   #generator=torch.Generator().manual_seed(seed)
                                   )
    print('Train on {}'.format(config.train_split))
    total_step = len(train_loader)

    min_lr_head = 5e-5
    min_lr_backbone = 1e-7

    restart_epochs = 1

    schedule_backbone = cosine_scheduler(config.backbone_lr, min_lr_backbone, 2, config.epoches, total_step)
    schedule_head = cosine_scheduler(config.head_lr, min_lr_head , 4, config.epoches, total_step)

    val_loader = get_video_dataset(config.val_split)
    val_loader = data.DataLoader(dataset=val_loader,
                                   batch_size=1,
                                   shuffle=False,
                                   num_workers=4, #intitial era 4
                                   pin_memory=False,
                                   worker_init_fn=seed_worker
                                   # generator=torch.Generator().manual_seed(seed + 1000)
                                   )

    # logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")
    print("Network-Train")
    logging.info('Config: epoch: {}; head_lr: {}; backbone_lr: {}; head_weight_decay: {}; backbone_weight_decay: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; '
                 'save_path: {}; decay_epoch: {}; min_lr_head: {}; min_lr_backbone: {}; restart_epochs: {}'.format(config.epoches, config.head_lr, config.backbone_lr, config.head_weight_decay, config.backbone_weight_decay, config.batchsize, config.size, config.clip,
                                                         config.decay_rate, config.save_path, config.decay_epoch, min_lr_head, min_lr_backbone, restart_epochs))
    print('Config: epoch: {}; head_lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; '
                 'save_path: {}; decay_epoch: {}'.format(config.epoches, config.head_lr, config.batchsize, config.size, config.clip,
                                                         config.decay_rate, config.save_path, config.decay_epoch))
    step = 0

    print("Start train...")

    start_epoch = 0

    if config.checkpoint_path != '':
        start_epoch = load_checkpoint(model, optimizer, config.checkpoint_path)

    for epoch in range(start_epoch, config.epoches):
        train(train_loader, model, optimizer, epoch, save_path, loss_func)
        val(val_loader, model, epoch, loss_func_val)

