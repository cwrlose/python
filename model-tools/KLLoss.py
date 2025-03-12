"""
logit_scale = model.logit_scale.exp()
logits_per_image, logits_per_text = create_logits(image_embedding,text_embedding,logit_scale)#对数概率分布

ground_truth = torch.tensor(gen_label(list_id),dtype=image_embedding.dtype,device=device)#普通概率分布
loss_imgs = loss_img(logits_per_image,ground_truth)
loss_texts = loss_txt(logits_per_text,ground_truth)
total_loss = (loss_imgs + loss_texts)/2
"""

import torch.nn.functional as F
import torch.nn as nn

class KLLoss(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()
        print('=========using KL Loss=and has temperature and * bz==========')
        self.error_metric = error_metric

    def forward(self, prediction, label):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label * 10, 1)
        loss = self.error_metric(probs1, probs2) * batch_size
        return loss
