import torch 


def one_hot(target,num_class):
    one_hot = torch.zeros(target.shape[0],num_class).cuda()
    one_hot = one_hot.scatter(1,target.long().view(-1,1),1.)
    return one_hot