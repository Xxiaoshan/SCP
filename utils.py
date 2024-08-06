import os

def ensure_path(new_path):
    if os.path.exists(new_path):
        pass
    else:
        os.makedirs(new_path)
        print('创建新路径:{}'.format(new_path))

def update_param(model, pretrained_dict):#pretrained_dict是预训练的权值
    model_dict = model.state_dict()#当前模型加入了多头注意力的，原先是127个权值，现在7个是多头注意力的权值，一共134个权值
    pretrained_dict = {k: v for k, v in pretrained_dict.items()}
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}#如果预训练模型参数名字在当前模型的里也有，那就用预训练模型的参数代替它。
    model_dict.update(pretrained_dict)  #覆盖现有的字典里的条目    #用预训练得到的权值更新现在网络的权值，只更新全面的部分，因为当前cec模型加入了注意力机制
    model.load_state_dict(model_dict)
    return model