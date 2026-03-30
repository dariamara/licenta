import math
import numpy as np

def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_backbone_lr, init_head_lr, epoch, decay_rate=0.1, decay_epoch=30, warmup_epochs=0):
    if epoch < warmup_epochs:
        backbone_lr = init_backbone_lr * (epoch + 1) / warmup_epochs
        head_lr = init_head_lr * (epoch + 1) / warmup_epochs
    else:
        decay = decay_rate ** ((epoch - warmup_epochs) // decay_epoch)
        backbone_lr = decay * init_backbone_lr
        head_lr = decay * init_head_lr
    for param_group in optimizer.param_groups:
        if param_group['name'] == 'backbone_params':
            param_group['lr'] = backbone_lr
        elif param_group['name'] == 'head_params':
            param_group['lr'] = head_lr
        lr = param_group['lr']
    return lr

def adjust_lr_step(optimizer, lr, name):
    for param_group in optimizer.param_groups:
        if param_group['name'] == name:
            param_group['lr'] = lr

def cosine_scheduler(base_value, final_value, epochs, total_epoches, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1, restart_epochs=1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    final_schedule = warmup_schedule

    for _ in range(restart_epochs):
        iters = np.arange((epochs * niter_per_ep - warmup_iters) // restart_epochs)
        schedule = np.array(
            [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])
        final_schedule = np.concatenate((final_schedule, schedule))

    if len(final_schedule) < total_epoches * niter_per_ep:
        final_schedule = np.concatenate((final_schedule, (np.full(total_epoches * niter_per_ep - len(final_schedule), final_schedule[-1]))))
    return final_schedule
