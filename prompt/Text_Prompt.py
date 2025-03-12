"""
class 代表类别

num_text_aug 插入长度

text_dict：每个键对应的值是一个张量，包含了该文本提示模板与所有动作类别名称组合后的分词文本序列。
这样，text_dict 就包含了所有文本提示模板与动作类别名称组合后的分词文本序列的集合
"""
import torch
import clip

def text_prompt(data):
    text_aug = [f"a photo of action {{}}", f"a picture of action {{}}", f"Human action of {{}}", f"{{}}, an action",
                f"{{}} this is an action", f"{{}}, a video of action", f"Playing action of {{}}", f"{{}}",
                f"Playing a kind of action, {{}}", f"Doing a kind of action, {{}}", f"Look, the human is {{}}",
                f"Can you recognize the action of {{}}?", f"Video classification of {{}}", f"A video of {{}}",
                f"The man is {{}}", f"The woman is {{}}"]
    text_dict = {}
    num_text_aug = len(text_aug)

    for ii, txt in enumerate(text_aug):
        text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for i, c in data.classes])

    classes = torch.cat([v for k, v in text_dict.items()])

    return classes, num_text_aug,text_dict

"""
for kkk,(images,list_id) in enumerate(tqdm(train_loader)):
text_id = numpy.random.randint(num_text_aug,size=len(list_id))#类别长度 上限
#text_dict[j][i, :]：根据 text_id 中的索引 j 从 text_dict 中选择对应的编码张量，再根据 list_id 中的索引 i 从该张量中选择一行。
texts = torch.stack([text_dict[j][i,:] for i,j in zip(list_id,text_id)]
"""
