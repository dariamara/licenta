import argparse


parser = argparse.ArgumentParser()

# optimizer
parser.add_argument('--gpu_id', type=str, default='0, 1', help='train use gpu')
parser.add_argument('--lr_mode', type=str, default="poly")
parser.add_argument('--backbone_lr', type=float, default=3e-5)
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
#contributie
# Changed from (256, 448) to (224, 448): Swin window_size=7 requires the patch
# grid dimensions to be divisible by 7. With height=224 the patch grid is [56,112]
# and all 4 stage resolutions ([56,112],[28,56],[14,28],[7,14]) divide evenly by 7.
# With height=256 the grid is [64,112] where 64 % 7 != 0, causing a runtime crash.
parser.add_argument('--size', type=tuple,
                    default=(224, 448))
#contributie
parser.add_argument('--batchsize', type=int, default=24)
parser.add_argument('--video_time_clips', type=int, default=6)

parser.add_argument('--save_path', type=str, default='/tmp/work/')

parser.add_argument('--augment', action='store_true', help='Enable augmentation')

parser.add_argument('--checkpoint_path', type=str, default='', help='Load training state')

parser.add_argument('--pth_path', type=str, default='/tmp/work/epoch_1/PNSPlus.pth')

parser.add_argument('--test_split', type=str, default="TestEasyDataset/Unseen")

parser.add_argument('--val_split', type=str, default="TestEasyDataset/Seen")

parser.add_argument('--ema_test', action='store_true', help='Enable EMA for test')

parser.add_argument('--ema_train', action='store_true', help='Enable EMA for train')

parser.add_argument('--save_path_preds', type=str, default="/storage/datasets/preds")

parser.add_argument('--use_kan', action='store_true', help='Use KAN blocks')

#contributie
parser.add_argument('--swin_pretrained', type=str, default='',
                    help='Path to Swin-B ImageNet-22k pretrained checkpoint '
                         '(swin_base_patch4_window7_224_22k.pth)')
#contributie

config = parser.parse_args()