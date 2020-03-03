#lr_scheduler(optimizer, i, **schedule_param)

def inv_lr_scheduler(optimizer, iter_num, gamma, power, lr=0.001, weight_decay=0.0005):
    lr = lr * (1 + gamma * iter_num)**(-power)
    i=0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = weight_decay * param_group['decay_mult']
        i+=1
    return optimizer

schedule_dict = {"inv" : inv_lr_scheduler}