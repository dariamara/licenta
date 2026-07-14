import os
from functools import lru_cache

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

from lib.module.PNSPlusNetworkSegNeXt import PNSNet

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
PTH_PATH = os.path.join(WEIGHTS_DIR, "SegNeXt_PNSPlus.pth")
STATISTICS_PATH = os.path.join(WEIGHTS_DIR, "statistics.pth")

SIZE = (256, 448)  # (height, width) -- matches config.size this checkpoint was trained with
TIME_CLIPS = 6  # matches config.video_time_clips
DEVICE = torch.device("cpu")

OVERLAY_COLOR = (0, 255, 0)  # green
OVERLAY_ALPHA = 0.4


@lru_cache(maxsize=1)
def _load_model():
    # use_kan=True: up_sample_high is a 4x4 transpose conv in this checkpoint (2x2 is what
    # use_kan=False builds), so this SegNeXt checkpoint was trained with the KAN blocks on.
    model = PNSNet(bn_out=(SIZE[0] // 8, SIZE[1] // 8), use_kan=True)
    checkpoint = torch.load(PTH_PATH, map_location=DEVICE, weights_only=False)
    # training wrapped the model in nn.DataParallel, which prefixes every key with "module."
    state_dict = {k.replace("module.", "", 1): v for k, v in checkpoint["model_state_dict"].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


@lru_cache(maxsize=1)
def _load_statistics():
    stats = torch.load(STATISTICS_PATH, map_location=DEVICE, weights_only=False)
    return stats["mean"], stats["std"]


def _preprocess(pil_img, mean, std):
    # Faithfully reproduces preprocess.py's Resize_video + toTensor_video + Normalize_video
    # pipeline so the model sees exactly what it saw during training. Normalize_video runs
    # *after* toTensor_video, so it normalizes tensor[:, :, i] -- i.e. the first 3 columns of
    # the (C, H, W) tensor, not the 3 channels. That's a quirk inherited from the training
    # code, not a fix applied here: reproducing it is required to match the trained weights.
    resized = pil_img.convert("RGB").resize((SIZE[1], SIZE[0]), Image.BILINEAR)
    tensor = ToTensor()(resized)
    for i in range(3):
        tensor[:, :, i] -= float(mean[i])
    for i in range(3):
        tensor[:, :, i] /= float(std[i])
    return tensor


def process_frames(input_paths, output_paths):
    """
    Runs PNSNet (SegNeXt) video-polyp-segmentation over a sequence of consecutive frames.

    The network is temporal: every inference clip is [anchor_frame, 6 consecutive local
    frames], and only the 6 local frames receive a predicted mask -- mirroring the test-time
    windowing in dataloader.py's VideoDataset (the very first uploaded frame only ever serves
    as the anchor, and never gets a direct prediction of its own).
    """
    assert len(input_paths) == len(output_paths)
    model = _load_model()
    mean, std = _load_statistics()

    n = len(input_paths)
    frames = [Image.open(p) for p in input_paths]

    # the windowing logic below assumes at least TIME_CLIPS+1 frames; pad short uploads by
    # repeating the last frame so a single clip can still be built
    padded_frames = list(frames)
    while len(padded_frames) < TIME_CLIPS + 1:
        padded_frames.append(padded_frames[-1])
    padded_n = len(padded_frames)

    tensors = [_preprocess(f, mean, std) for f in padded_frames]
    masks = [None] * padded_n  # frame index -> predicted probability mask (H, W); None = anchor-only

    begin = 1
    with torch.no_grad():
        while begin < padded_n:
            if padded_n - begin - 1 < TIME_CLIPS:
                begin = padded_n - TIME_CLIPS
            clip_indices = [0] + [begin + t for t in range(TIME_CLIPS)]
            clip = torch.stack([tensors[i] for i in clip_indices], dim=0).unsqueeze(0)  # (1, 7, 3, H, W)
            pred = model(clip)  # (TIME_CLIPS, 1, H, W)
            for t in range(TIME_CLIPS):
                masks[begin + t] = pred[t, 0]
            begin += TIME_CLIPS

    for i in range(n):
        mask = masks[i]
        frame = frames[i].convert("RGB")
        if mask is None:
            # anchor frame: no direct prediction under this scheme, show the original frame untouched
            out = frame
        else:
            prob = mask.unsqueeze(0).unsqueeze(0)
            prob = F.interpolate(prob, size=(frame.height, frame.width), mode="bilinear", align_corners=False)
            binary_mask = (prob.squeeze(0).squeeze(0) > 0.5).float()
            # alpha_mask values are 0 or OVERLAY_ALPHA*255, so Image.composite blends the
            # color layer over the original frame only where the model predicted positive
            alpha_mask = ToPILImage()(binary_mask * OVERLAY_ALPHA)
            color_layer = Image.new("RGB", frame.size, OVERLAY_COLOR)
            out = Image.composite(color_layer, frame, alpha_mask)
        out.save(output_paths[i])
