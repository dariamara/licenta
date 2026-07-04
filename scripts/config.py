import argparse


parser = argparse.ArgumentParser()

# optimizer
parser.add_argument('--gpu_id', type=str, default='0, 1', help='train use gpu')
parser.add_argument('--lr_mode', type=str, default="poly")
# contributie start
# MedNeXt trains from scratch (no pretrained weights), so backbone needs a higher LR than ConvNeXt's 3e-5.
# Reduced from 2e-4 to 1e-4: fp16 activations in the 1024-channel bottleneck overflow with too high an LR.
parser.add_argument('--backbone_lr', type=float, default=1e-4)
# contributie end
parser.add_argument('--head_lr', type=float, default=1e-3)
parser.add_argument('--backbone_weight_decay', type=float, default=1e-8)
parser.add_argument('--head_weight_decay', type=float, default=1e-8)
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=16, help='every n epochs decay learning rate')
parser.add_argument('--clip', type=float, default=1.05, help='gradient clipping margin')

# train schedule
parser.add_argument('--epoches', type=int, default=15)

# data
parser.add_argument('--data_statistics', type=str,
                    default="/code/lib/dataloader/statistics.pth", help='The normalization statistics.')
parser.add_argument('--train_split', type=str,
                    default="TrainDataset")
parser.add_argument('--dataset_root', type=str,
                    default="/storage/datasets/SUN/data")
parser.add_argument('--size', type=tuple,
                    default=(256, 448))
parser.add_argument('--batchsize', type=int, default=24)
parser.add_argument('--video_time_clips', type=int, default=6)

parser.add_argument('--save_path', type=str, default='/tmp/work/')

parser.add_argument('--augment', action='store_true', help='Enable augmentation')

parser.add_argument('--checkpoint_path', type=str, default='', help='Load training state')

parser.add_argument('--pth_path', type=str, default='/tmp/work/epoch_15/PNSPlus.pth')

parser.add_argument('--test_split', type=str, default="TestHardDataset/Unseen")

parser.add_argument('--val_split', type=str, default="TestHardDataset/Seen")

parser.add_argument('--ema_test', action='store_true', help='Enable EMA for test')

parser.add_argument('--ema_train', action='store_true', help='Enable EMA for train')

parser.add_argument('--save_path_preds', type=str, default="/storage/datasets/preds")

parser.add_argument('--use_kan', action='store_true', help='Use KAN blocks')
parser.add_argument('--no_kan', action='store_true', help='Ablation: swap KAN spline layers for plain linear layers')
parser.add_argument('--kan_depths', type=int, nargs=2, default=(1, 1), help='Number of KAN blocks per stage')
parser.add_argument('--drop_path_rate', type=float, default=0.0, help='Stochastic depth rate across KAN blocks')

config = parser.parse_args()