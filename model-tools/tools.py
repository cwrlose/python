
import numpy
"""
gen_label 函数的主要功能是根据输入的标签列表 labels 生成一个二维的二进制矩阵 gt（通常可作为某种任务中的标签矩阵）。这个矩阵反映了标签之间的相等关系，
即矩阵中第 i 行第 k 列的元素为 1 表示第 i 个标签和第 k 个标签相同，为 0 则表示不同。
"""
def gen_label(labels):
    num = len(labels)
    gt = numpy.zeros(shape=(num,num))
    for i, label in enumerate(labels):
        for k in range(num):
            if labels[k] == label:
                gt[i,k] = 1
    return gt

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()

def convert_models_to_fp16(model):
    print(model)
    for p in model.parameters():
        p.data = p.data.half()
        p.grad.data = p.grad.data.half()

"""
create_logits 函数的主要目的是计算两个输入张量 x1 和 x2 之间的余弦相似度，并将其乘以一个缩放因子 logit_scale 得到 logits 值。在机器学习中，
logits 通常是模型输出的未经过激活函数处理的原始分数，后续可能会用于计算损失或进行分类等操作。
"""
"""
也是一个二维矩阵
"""
def create_logits(x1, x2, logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 = logit_scale * x1 @ x2.t()
    logits_per_x2 = logit_scale * x2 @ x1.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2
