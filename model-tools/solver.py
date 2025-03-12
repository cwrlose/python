
import torch.optim as optim
from utils.lr_scheduler import WarmupMultiStepLR, WarmupCosineAnnealingLR
"""
选择优化器
optimizer = _optimizer(config, model, fusion_model)
"""
def _optimizer(config, model, fusion_model):
    if config.solver.optim == 'adam':
        optimizer = optim.Adam([{'params': model.parameters()},  
         {'params': fusion_model.parameters(), 'lr': config.solver.lr * config.solver.f_ratio}],
                               lr=config.solver.lr, betas=(0.9, 0.98), eps=1e-8,
                               weight_decay=0.2)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
        print('Adam')
    elif config.solver.optim == 'sgd':

        optimizer = optim.SGD([{'params': model.parameters()},  
         {'params': fusion_model.parameters(), 'lr': config.solver.lr * config.solver.f_ratio}],
                              config.solver.lr,
                              momentum=config.solver.momentum,
                              weight_decay=config.solver.weight_decay)
        print('SGD')
    elif config.solver.optim == 'adamw':
        vision_params = list(map(id, model.visual.parameters()))
        text_params = filter(lambda p: id(p) not in vision_params,
                             model.parameters())

        optimizer = optim.AdamW([{'params': text_params},
                                 {'params': model.visual.parameters(), 'lr': config.solver.lr * config.solver.ratio},
                                 {'params': fusion_model.parameters(), 'lr': config.solver.lr * config.solver.f_ratio}],
                                betas=(0.9, 0.98), lr=config.solver.lr, eps=1e-8,
                                weight_decay=config.solver.weight_decay)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        print('AdamW')
    else:
        raise ValueError('Unknown optimizer: {}'.format(config.solver.optim))
    return optimizer
"""
选择学习率策略 （选余弦）
WarmupCosineAnnealingLR：这是一个带有热身阶段的余弦退火学习率调度器。在热身阶段，
学习率会从一个较小的值逐步增加到初始学习率；之后，学习率会按照余弦函数的规律逐渐减小，直到训练结束
config.solver.epochs：这是训练的总轮数。余弦退火策略会根据这个总轮数来调整学习率，使得学习率在整个训练过程中逐渐降低。
warmup_epochs=config.solver.lr_warmup_step：这是热身阶段的轮数。在训练的前 config.solver.lr_warmup_step 轮中，
学习率会从一个较小的值逐渐增加到初始学习率
"""

def _lr_scheduler(config,optimizer):
    if config.solver.type == 'cosine':
        lr_scheduler = WarmupCosineAnnealingLR(
            optimizer,
            config.solver.epochs,
            warmup_epochs=config.solver.lr_warmup_step
        )
    elif config.solver.type == 'multistep':
        if isinstance(config.solver.lr_decay_step, list):
            milestones = config.solver.lr_decay_step
        elif isinstance(config.solver.lr_decay_step, int):
            milestones = [
                config.solver.lr_decay_step * (i + 1)
                for i in range(config.solver.epochs //
                               config.solver.lr_decay_step)]
        else:
            raise ValueError("error learning rate decay step: {}".format(type(config.solver.lr_decay_step)))
        lr_scheduler = WarmupMultiStepLR(
            optimizer,
            milestones,
            warmup_epochs=config.solver.lr_warmup_step
        )
    else:
        raise ValueError('Unknown lr scheduler: {}'.format(config.solver.type))
    return lr_scheduler
